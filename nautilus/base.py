# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
from distutils.version import LooseVersion
import tensorflow as tf
import datetime
import numpy as np
import shutil
import time
import copy
import networkx as nx
import json
import operator
import gzip
import itertools

import multiprocessing
multiprocessing.set_start_method('forkserver', force=True)

from shutil import copyfile
from . import constants
from .utils import pretty_print, is_valid_evaluation_metric, get_dtype_size, get_tf_dtype_from_np_dtype, is_lower_evaluation_metric_better
from .utils import get_model_sub_graph_from_materialized_layers, check_layer_configs_match, merge_model_metadata_graphs, get_compute_cost
from .keras import NautilusOptimizer, NautilusModel
from .keras import intermediate_features_mat_fn, sub_graph_model_gen_fn, model_train_fn, gen_trained_model_with_weights
from .profiling import model_initialization_fn, estimate_model_memory_consumption, estimate_model_flops_consumption
from .optimizer import get_materialization_points, get_sub_model_and_used_layers, find_execution_plan


class ModelSelection(object):
    """Nautilus model search base class"""

    def __init__(self, estimator_gen_fn, search_space, num_params, evaluation_metric, label_columns,
                 feature_columns, max_num_records, storage_path, storage_budget, memory_budget, workspace_memory,
                 compute_throughput, disk_throughput, shuffle_buffer_size, custom_objects, extra_configs, verbose):
        if is_valid_evaluation_metric(evaluation_metric):
            self.evaluation_metric = evaluation_metric
        else:
            raise Exception(
                'Unsupported evaluation metric: {}'.format(evaluation_metric))

        self.max_num_records = max_num_records
        self.estimator_gen_fn = estimator_gen_fn
        self.num_params = num_params

        assert len(label_columns) == 1, 'Nautilus currently support only single output models.'
        self.label_cols = label_columns
        
        self.feature_cols = feature_columns
        self.custom_objects = custom_objects
        self.extra_configs = extra_configs
        self.verbose = verbose
        self.storage_path = storage_path
        self.storage_budget = storage_budget
        self.memory_budget = memory_budget
        self.workspace_memory = workspace_memory
        self.compute_throughput = float(compute_throughput) * 1e12 # TFLOPS => FLOPS
        self.disk_throughput = disk_throughput
        self.shuffle_buffer_size = shuffle_buffer_size
        self.estimator_param_maps = {}
        self.estimator_param_types = {}
        self.estimator_batch_sizes = {}
        self.estimator_num_epochs = {}
        self.estimator_losses = {}
        self.estimator_optimizers = {}
        self.estimator_metrics = {}
        self.estimator_training_history = {}

        self.cycle = 0
        self.num_train_records = 0
        self.num_valid_records = 0

        # Various metadata graphs populated during analysis
        self.super_model_graph = None
        self.optimized_model_meta_graphs = None
        self.materialized_layers = None

        gpu_physical_devices = tf.config.list_physical_devices('GPU')
        # TODO: Relax this condition to support multiple GPUs
        assert len(gpu_physical_devices) <= 1, "At most 1 GPU is supported."
        try:
            tf.config.experimental.set_memory_growth(
                gpu_physical_devices[0], True)
        except:
            # Invalid device or cannot modify virtual devices once initialized.
            pass

        self.search_space = search_space
        # validate the search space
        self._validate_search_space()

        self.optimized_training_super_models = None

        self.estimator_param_maps, self.estimator_param_types = self._generate_all_param_maps()
        self._gen_model_selection_plan()

        
    def get_path_to_best_model(self, a_cycle=None):
        """Returns the path to the best model of the specified active learning cycle.

        Returns:
            string: File system path to the model checkpoint.
        """
        if a_cycle is None and self.cycle == 0:
            return None
        else:
            if a_cycle is not None:
                assert a_cycle > 0 and a_cycle <= self.cycle, 'Invalid cycle number: {}. Cycle number has to be between 1 amd {}'.format(a_cycle, self.cycle)
            else:
                a_cycle = self.cycle
            return os.path.join(self.storage_path, 'models', 'best_models', '{}.h5'.format(a_cycle))


    def fit(self, X_train, y_train, X_valid=None, y_valid=None, incremental=True):
        """
        Execute the model selection/AutoML workload on the given DataFrame.

        :param X_train: Training data dictionary containing numpy arrays.
        :param y_train: Training data labels.
        :param X_train: Validation data dictionary containing numpy arrays.
        :param y_train: Validation data labels.
        :param incremental: Whether to reuse any previously fitted training and validation data.
        :return: tf.keras.Model
        """

        assert X_train is not None and y_train is not None
        data_path_root = os.path.join(self.storage_path, 'data')
        self.estimator_training_history = {}

        if not incremental:
            self.num_train_records = 0
            self.num_valid_records = 0
            self.cycle = 0

            if os.path.exists(data_path_root) and os.path.isdir(data_path_root):
                shutil.rmtree(data_path_root)
            os.makedirs(data_path_root)
            os.makedirs(os.path.join(data_path_root, 'train'))
            os.makedirs(os.path.join(data_path_root, 'valid'))

        #---------------------------------- 1. Update the training data ------------------------------------#
        if self.verbose > 0:
            ct = datetime.datetime.now()
            begin_time = ct
            print('NAUTILUS=>{}: Materializing training data'.format(ct))
        Ds = [X_train, y_train]
        splits = ['train', 'train']
        self.num_train_records += X_train[list(X_train.keys())[0]].shape[0]

        if X_valid is not None and y_valid is not None:
            Ds.extend([X_valid, y_valid])
            splits.extend(['valid', 'valid'])
            self.num_valid_records += X_valid[list(X_valid.keys())[0]].shape[0]

        for D, split in zip(Ds, splits):
            for k in D:
                data_path = os.path.join(
                    data_path_root, split, '{}.npy'.format(k))
                num_records = D[k].shape[0]
                with open(data_path, 'ab') as f:
                    for i in range(num_records):
                        np.save(f, D[k][i:i+1])
        if self.verbose > 0:
            ct = datetime.datetime.now()
            print('NAUTILUS=>{}: Completed materializing training data. Elapsed time: {}'.format(ct, ct - begin_time))

        #-------------------- 2. Incrementally materialize the intermediate features. ---------------------#
        if constants.USE_MATERIALIZATION_OPTIMIZATION not in self.extra_configs or self.extra_configs[constants.USE_MATERIALIZATION_OPTIMIZATION] \
            and self.materialized_layers is not None and len(self.materialized_layers) > 0:
            if self.verbose > 0:
                ct = datetime.datetime.now()
                begin_time = ct
                print('NAUTILUS=>{}: Generating intermediate features'.format(ct))
            predict_batch_size = max(self.estimator_batch_sizes.values())
            p = multiprocessing.Process(target=intermediate_features_mat_fn, args=(X_train, X_valid, self.storage_path, predict_batch_size, self.memory_budget, self.custom_objects))
            p.start()
            p.join()

            if self.verbose > 0:
                ct = datetime.datetime.now()
                print('NAUTILUS=>{}: Completed generating intermediate features. Elapsed time: {}'.format(ct, ct - begin_time))

        #-------------------------------------- 3. Train models. -----------------------------------------#
        if self.verbose > 0:
            ct = datetime.datetime.now()
            begin_time = ct
            print('NAUTILUS=>{}: Starting model training'.format(ct))

        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        if self.optimized_training_super_models is not None:
            for k in self.optimized_training_super_models:
                batch_size = self.optimized_training_super_models[k]['batch_size']
                num_epochs = self.optimized_training_super_models[k]['num_epochs']
                source_models = self.optimized_training_super_models[k]['models']
                losses = [self.estimator_losses[orig_model_name] for orig_model_name in source_models]
                metrics = [self.estimator_metrics[orig_model_name] for orig_model_name in source_models]
                metrics = list(itertools.chain(*metrics))
                optimizers = [self.estimator_optimizers[orig_model_name] for orig_model_name in self.optimized_training_super_models[k]['models']]
                # label_cols = self.label_cols * len(optimizers)

                p = multiprocessing.Process(target=model_train_fn, args=(
                            str(k), source_models, losses, optimizers, metrics, self.storage_path, self.feature_cols, self.label_cols, batch_size, num_epochs,
                            self.memory_budget, self.shuffle_buffer_size, return_dict, self.custom_objects))
                p.start()
                p.join()
            
                self.estimator_training_history.update(return_dict['train_history'])
        else:
            # No optimizations.
            for k in self.estimator_param_maps:
                batch_size = self.estimator_batch_sizes[k]
                num_epochs = [self.estimator_num_epochs[k]]
                losses = [self.estimator_losses[k]]
                optimizers = [self.estimator_optimizers[k]]
                metrics = self.estimator_metrics[k]

                p = multiprocessing.Process(target=model_train_fn, args=(str(k), [k], losses, optimizers, metrics, self.storage_path, self.feature_cols,
                    self.label_cols, batch_size, num_epochs, self.memory_budget, self.shuffle_buffer_size, return_dict, self.custom_objects))
                p.start()
                p.join()
            
                self.estimator_training_history.update(return_dict['train_history'])


        self.cycle += 1
        if self.verbose > 0:
            ct = datetime.datetime.now()
            print('NAUTILUS=>{}: Completed model training. Elapsed time: {}'.format(ct, ct - begin_time))
            for m in self.estimator_training_history:
                for k in self.estimator_training_history[m]:
                    print('NAUTILUS=>{}: Cycle: {}, Model: {}, {}: {}'.format(ct, self.cycle, m, k, self.estimator_training_history[m][k]))

        # Return best trained model.
        if self.verbose > 0:
            ct = datetime.datetime.now()
            begin_time = ct
            print('NAUTILUS=>{}: Starting to generating the best model'.format(ct))
        
        best_model = self._get_best_trained_model()

        if self.verbose > 0:
            ct = datetime.datetime.now()
            print('NAUTILUS=>{}: Completed generating the best model. Elapsed time: {}'.format(ct, ct - begin_time))
            print('NAUTILUS=>{}: Cycle: {}, Best Model: {}'.format(ct, self.cycle, best_model))
            for k in self.estimator_training_history[best_model]:
                print('NAUTILUS=>{}: Cycle: {}, Best model {}: {}'.format(ct, self.cycle, k, self.estimator_training_history[best_model][k]))


    def _get_best_trained_model(self):
        comparator = lambda x,y: x > y if is_lower_evaluation_metric_better(self.evaluation_metric) else x < y
        best_model = None
        best_model_eval_metric = np.inf if is_lower_evaluation_metric_better(self.evaluation_metric) else -np.inf

        for k in self.estimator_training_history:
            eval_val = self.estimator_training_history[k]['val_{}'.format(self.evaluation_metric)][-1]
            if comparator(best_model_eval_metric, eval_val):
                best_model_eval_metric = eval_val
                best_model = k

        assert best_model is not None
        optimized_model_name = None
        if self.optimized_training_super_models is not None:
            for k in self.optimized_training_super_models:
                source_models = self.optimized_training_super_models[k]['models']
                if best_model in source_models:
                    optimized_model_name = k
                    break
        else:
            optimized_model_name = best_model

        use_optimized = self.optimized_training_super_models is not None
        p = multiprocessing.Process(target=gen_trained_model_with_weights, args=(self.storage_path, best_model, optimized_model_name, self.estimator_losses[best_model],
            self.estimator_metrics[best_model], use_optimized,  self.cycle, self.memory_budget, self.custom_objects))
        p.start()
        p.join()
    
        return best_model


    def _gen_model_selection_plan(self):
        #----------------------------------------- 1. Model Profiling ---------------------------------------------
        self._create_storage_sub_dirs()
        if self.verbose > 0:
            ct = datetime.datetime.now()
            begin_time = datetime.datetime.now()
            print('NAUTILUS=>{}: Starting model profiling'.format(ct))

        benchmark = (constants.USE_MATERIALIZATION_OPTIMIZATION not in self.extra_configs or self.extra_configs[constants.USE_MATERIALIZATION_OPTIMIZATION]) \
                    or (constants.USE_MODEL_MERGE_OPTIMIZATION not in self.extra_configs or self.extra_configs[constants.USE_MODEL_MERGE_OPTIMIZATION])

        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        for k in self.estimator_param_maps:
            p = multiprocessing.Process(
                target=model_initialization_fn, args=(str(k), self.storage_path, self.estimator_gen_fn, self.estimator_param_maps[k], self.estimator_param_types, benchmark,
                self.custom_objects, return_dict)
            )
            p.start()
            p.join()
        metadata_values = return_dict

        for k in self.estimator_param_maps:
            self.estimator_batch_sizes[k] = metadata_values[k]['batch_size']
            self.estimator_num_epochs[k] = metadata_values[k]['num_epochs']
            self.estimator_losses[k] = metadata_values[k]['loss']
            self.estimator_optimizers[k] = metadata_values[k]['optimizer']
            self.estimator_metrics[k] = metadata_values[k]['metrics']

        if self.verbose > 0:
            ct = datetime.datetime.now()
            print('NAUTILUS=>{}: Completed model profiling. Elapsed time: {}'.format(ct, ct - begin_time))
            print('NAUTILUS=>{}: Model parameters'.format(ct))
            pretty_print(self.estimator_param_maps)
        #-----------------------------------------------------------------------------------------------------------


        if (constants.USE_MATERIALIZATION_OPTIMIZATION not in self.extra_configs or self.extra_configs[constants.USE_MATERIALIZATION_OPTIMIZATION]) \
            or (constants.USE_MODEL_MERGE_OPTIMIZATION not in self.extra_configs or self.extra_configs[constants.USE_MODEL_MERGE_OPTIMIZATION]):
            
            #-------------------------------- 2 Optimized Model Training Plan Generation -----------------------------
            if self.verbose > 0:
                ct = datetime.datetime.now()
                begin_time = datetime.datetime.now()
                print('NAUTILUS=>{}: Generating optimized model selection plan'.format(ct))

            # Performing optimizations
            # Create informational super model
            self.super_model_graph = nx.DiGraph()
            for k in metadata_values:
                merge_model_metadata_graphs(self.super_model_graph, metadata_values[k]['metadata_graph'], k, self.estimator_num_epochs[k])

            # Generating feature materialization model and modified model graphs to reused materialized features
            global_input_names = [l for l in self.super_model_graph.nodes() if self.super_model_graph.in_degree(l) == 0]


            #------------------------------ 2.1 Materialization Optimization-------------------------------------------
            if constants.USE_MATERIALIZATION_OPTIMIZATION not in self.extra_configs or self.extra_configs[constants.USE_MATERIALIZATION_OPTIMIZATION]:
                materialized_layers_candidates, _ = get_materialization_points(self.super_model_graph, self.max_num_records, self.storage_budget,
                    self.compute_throughput, self.disk_throughput)
                if self.verbose > 0:
                    ct = datetime.datetime.now()
                    print('NAUTILUS=>{}: Materialized layers candidates: {}'.format(ct, ','.join(materialized_layers_candidates)))
            else:
                materialized_layers_candidates = []

            merge_candidates = {}
            for k in self.estimator_param_maps:
                # Finding which materialized layers should be reused
                output_layers = [n for n in self.super_model_graph.nodes() if self.super_model_graph.out_degree(n) == 0 and str(k) in self.super_model_graph.nodes()[n]['models']]
                G, all_used, used_materialized = get_sub_model_and_used_layers(self.super_model_graph, materialized_layers_candidates, output_layers, self.compute_throughput, self.disk_throughput, self.max_num_records)

                mem_consumption = estimate_model_memory_consumption(all_used, self.super_model_graph, self.estimator_batch_sizes[k], self.workspace_memory)
                compute_flops, load_flops = estimate_model_flops_consumption(all_used, used_materialized, G, self.compute_throughput, self.disk_throughput, self.max_num_records)
                flops_consumption = compute_flops + load_flops

                merge_candidates[str(k)] = {
                    'models': [str(k)],
                    'model_graph': G,
                    'output_layers': output_layers,
                    'flops_consumption': flops_consumption,
                    'compute_flops': compute_flops,
                    'load_flops': load_flops,
                    'memory_consumption': mem_consumption,
                    'used_materialized_layers': used_materialized,
                    'batch_size': self.estimator_batch_sizes[k],
                    'num_epochs': [self.estimator_num_epochs[k]]
                }

            #-------------------------------------- 2.2 Model Fusion Optimization-----------------------------------------
            if constants.USE_MODEL_MERGE_OPTIMIZATION not in self.extra_configs or self.extra_configs[constants.USE_MODEL_MERGE_OPTIMIZATION]:
                merge_candidates = find_execution_plan(self.super_model_graph, merge_candidates, materialized_layers_candidates, self.memory_budget, self.workspace_memory,
                    self.compute_throughput, self.disk_throughput, self.max_num_records)

            all_used_materialized_layers = []
            for k in merge_candidates:
                for l in merge_candidates[k]['used_materialized_layers']:
                    if l not in all_used_materialized_layers:
                        all_used_materialized_layers.append(l)

            if self.verbose > 0:
                ct = datetime.datetime.now()
                print('NAUTILUS=>{}: Used materialized layers: {}'.format(ct, ','.join(all_used_materialized_layers)))
                print('NAUTILUS=>{}: Model training plan: {}'.format(ct, json.dumps({m:{k: merge_candidates[m][k] for k in merge_candidates[m] if k != 'model_graph'} for m in merge_candidates})))
                pretty_print(merge_candidates)

            self.optimized_training_super_models = merge_candidates

            self.materialized_layers = all_used_materialized_layers
            if len(all_used_materialized_layers) > 0:
                model_gen_params = [['feature_mat', self.super_model_graph, global_input_names, all_used_materialized_layers]]
            else:
                model_gen_params = []

            for k in self.optimized_training_super_models:
                G = self.optimized_training_super_models[k]['model_graph']
                output_layers = self.optimized_training_super_models[k]['output_layers']
                used_materialized = self.optimized_training_super_models[k]['used_materialized_layers']
                model_gen_params.append([k, G, global_input_names + used_materialized, output_layers])

            if self.verbose > 0:
                ct = datetime.datetime.now()
                print('NAUTILUS=>{}: Completed generating optimized model selection plan. Elapsed time: {}'.format(ct, ct - begin_time))

            #---------------------------------------------------------------------------------------------------------------
            
            # Generating modified model graphs to reuse materialized features
            if self.verbose > 0:
                ct = datetime.datetime.now()
                begin_time = datetime.datetime.now()
                print('NAUTILUS=>{}: Generating fused models'.format(ct))

            return_dict = manager.dict()
            p = multiprocessing.Process(target=sub_graph_model_gen_fn, args=(model_gen_params, self.storage_path, self.custom_objects, self.memory_budget, return_dict))
            p.start()
            p.join()

            if self.verbose > 0:
                ct = datetime.datetime.now()
                print('NAUTILUS=>{}: Completed generating fused models. Elapsed time: {}'.format(ct, ct - begin_time))

                # Theoretical Speedup Calculation
                current_practice_flops = 0
                for k in metadata_values:
                    G = metadata_values[k]['metadata_graph']
                    current_practice_flops += sum([get_compute_cost(G, l, 1) * G.nodes()[l]['num_epochs'][0] for l in G.nodes()])
                
                print('NAUTILUS=>{}: Current Practice FLOPs: {}'.format(ct, current_practice_flops))

                optimal_flops = 0
                for l in self.super_model_graph.nodes():
                    if self.super_model_graph.in_degree(l) != 0:
                        if not self.super_model_graph.nodes()[l]['materializable']:
                            assert len(self.super_model_graph.nodes()[l]['num_epochs']) == 1
                            optimal_flops += get_compute_cost(self.super_model_graph, l, 1) * self.super_model_graph.nodes()[l]['num_epochs'][0]

                print('NAUTILUS=>{}: Optimal FLOPs: {}'.format(ct, optimal_flops))
                print('NAUTILUS=>{}: Theoretical Speedup: {}'.format(ct, current_practice_flops/optimal_flops))


    def _create_storage_sub_dirs(self):
        if os.path.exists(self.storage_path) and os.path.isdir(self.storage_path):
            shutil.rmtree(self.storage_path)

        os.makedirs(self.storage_path)
        os.makedirs(os.path.join(self.storage_path, 'models'))
        os.makedirs(os.path.join(self.storage_path, 'data/train'))
        os.makedirs(os.path.join(self.storage_path, 'data/valid'))

    def _validate_search_space(self):
        raise NotImplementedError()

    def _generate_all_param_maps(self):
        raise NotImplementedError()
