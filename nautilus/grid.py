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

import itertools
import numpy as np
from .base import ModelSelection
from .hparams import _HP, _HPChoice


class GridSearch(ModelSelection):
    """Performs grid search using the given param grid

    :param estimator_gen_fn: A function which takes
     in a dictionary of parameters and returns a compiled Keras model.
    :param search_space: A dictionary object defining the parameter search space.
    :param evaluation_metric: Evaluation metric used to pick the best model (default "loss").
    :param label_columns: (Optional) A list containing the names of the label/output columns (default ['label']).
    :param feature_columns: (Optional) A list containing the names of the feature columns (default ['features']).
    :param max_num_records: (Optional) The maximum number records that will be fitted by the model (default 10,000)
    :param storage_path: (Optional) Directory used for storing data (including intermediate features) and model checkpoints.
    :param storage_budget: (Optional) Maximum storage budget for storing intermediate features in GBs (default 50 GBs).
    :param memory_budget: (Optional) Maximum memory budget to be used during model training (GPU memory for GPU training or DRAM memory for CPU training) in GBs (default 10 GBs).
    :param workspace_memory: (Optional) Size of workspace memory region to be used for performing layer computations in GBs (default 1 GB).
    :param compute_throughput: (Optional) Device compute throughput in terms of TFLOPs (default 1 TFLOPs).
    :param disk_throughput: (Optional) Disk read throughput in terms of MB/s (default 200 MB/s).
    :param shuffle_buffer_size: (Optional) Number of records in the TensorFlow shuffle buffer size (default 10000).
    :param custom_objects: (Optional) Dictionary containing custom python classes needed for serialization and deserialization,
    :param extra_configs: (Optional) Dictionary of extra configurations (default {}).
    :param verbose: Debug output verbosity (0-2). Defaults to 1.

    :return: : The best model in tf.keras.Model format based on validation data.
    """

    def __init__(self, estimator_gen_fn, search_space,
                 evaluation_metric='loss', label_columns=['label'], feature_columns=['features'], eval_metric='loss', max_num_records=10000, storage_path='./storage', storage_budget=50,
                 memory_budget=10, workspace_memory=1, compute_throughput=1, disk_throughput=200, shuffle_buffer_size=10000, custom_objects=None, extra_configs={}, verbose=1):
        super(GridSearch, self).__init__(estimator_gen_fn, search_space, -1, evaluation_metric, label_columns, 
              feature_columns, max_num_records, storage_path, storage_budget, memory_budget, workspace_memory, compute_throughput, disk_throughput, shuffle_buffer_size, custom_objects, extra_configs, verbose)

    def _validate_search_space(self):
        search_space = self.search_space
        if not type(search_space) == dict:
            raise Exception('Search space has to be type dict. Provided: {}'.format(type(search_space)))

        if not all([isinstance(k, str) for k in search_space.keys()]):
            raise Exception('Only string values are allowed for hyperparameter space keys.')

        if not all([isinstance(k, _HPChoice) for k in search_space.values()]):
            raise Exception('All hyperparameter space values has to be of type nautilus.tune.base._HPChoice.'
                            ' Nested search spaces are not supported yet')

    def _generate_all_param_maps(self):
        keys = self.search_space.keys()
        grid_values = [v.options for v in self.search_space.values()]
        param_types = {k : self.search_space[k].param_type for k in self.search_space}

        def _to_key_value_pairs(keys, values):
            # values = [v if isinstance(v, list) else v() for v in values]
            return [(key, value) for key, value in zip(keys, values)]

        return {str(i) : dict(_to_key_value_pairs(keys, prod)) for i, prod \
            in enumerate(itertools.product(*[v if isinstance(v, list) else v() for v in grid_values]))}, param_types


class RandomSearch(ModelSelection):
    """ Performs Random Search over the param grid

    :param estimator_gen_fn: A function which takes
     in a dictionary of parameters and returns a compiled tf.keras.Model.
    :param search_space: A dictionary object defining the parameter search space.
    :param num_params: Number of different training configurations to be trained.
    :param evaluation_metric: Evaluation metric used to pick the best model (default: "loss").
     defining the validation set. In the latter case the column value can be bool or int.
    :param label_columns: (Optional) A list containing the names of the label/output columns (default ['label']).
    :param feature_columns: (Optional) A list containing the names of the feature columns (default ['features']).
    :param max_num_records: (Optional) The maximum number records that will be fitted by the model (default 10,000)
    :param storage_path: (Optional) Directory used for storing data (including intermediate features) and model checkpoints.
    :param storage_budget: (Optional) Maximum storage budget for storing intermediate features in GBs (default 50 GBs).
    :param memory_budget: (Optional) Maximum memory budget to be used during model training (GPU memory for GPU training or DRAM memory for CPU training) in GBs (default 10 GBs).
    :param workspace_memory: (Optional) Size of workspace memory region to be used for performing layer computations in GBs (default 1 GB).
    :param compute_throughput: (Optional) Device compute throughput in terms of TFLOPs (default 1 TFLOPs).
    :param disk_throughput: (Optional) Disk read throughput in terms of MB/s (default 200 MB/s).
    :param shuffle_buffer_size: (Optional) Number of records in the TensorFlow shuffle buffer size (default 10000).
    :param custom_objects: (Optional) Dictionary containing custom python classes needed for serialization and deserialization.
    :param extra_configs: (Optional) Dictionary of extra configurations (default {}).
    :param verbose: Debug output verbosity (0-2). Defaults to 1.

    :return: : The best model in tf.keras.Model format based on validation data.
    """

    def __init__(self, estimator_gen_fn, search_space, num_models, num_params, evaluation_metric='loss',
                        label_columns=['label'], feature_columns=['features'], max_num_records=10000, storage_path='./storage', storage_budget=50, memory_budget=10, workspace_memory=1,
                        compute_throughput=1, disk_throughput=200, shuffle_buffer_size=10000, custom_objects=None, extra_configs={}, verbose=1):
        super(RandomSearch, self).__init__(estimator_gen_fn, evaluation_metric, search_space, num_params,
                                           label_columns, feature_columns, max_num_records, storage_path, storage_budget, memory_budget, workspace_memory,
                                           compute_throughput, disk_throughput, shuffle_buffer_size, custom_objects, extra_configs, verbose)

    def _validate_search_space(self):
        search_space = self.search_space
        if not type(search_space) == dict:
            raise Exception('Search space has to be type dict. Provided: {}'.format(type(search_space)))

        if not all([isinstance(k, str) for k in search_space.keys()]):
            raise Exception('Only string values are allowed for hyperparameter space keys.')

        if not all([isinstance(k, _HP) for k in search_space.values()]):
            raise Exception('All hyperparameter space values has to be of type nautilus.tune.base._HP.'
                            ' Nested search spaces are not supported yet')

    def _generate_all_param_maps(self):
        params = []
        keys = self.search_space.keys()
        param_types = {k : self.search_space[k].param_type for k in self.search_space}
        for _ in range(self.num_params):
            param_dict = {}
            for k in keys:
                param_dict[k] = self.search_space[k].sample_value()
            params.append(param_dict)
        return {str(i) : p for i, p in enumerate(params)}, param_types
