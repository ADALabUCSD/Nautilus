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
import networkx as nx
import tensorflow as tf
import numpy as np
from tensorflow.python.util import nest
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.ops import gradients
from tensorflow.python.ops import array_ops, math_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.distribute import distribution_strategy_context as ds_context

from tensorflow.python.keras.losses import LossFunctionWrapper
from .utils import get_model_sub_graph_from_materialized_layers, get_tf_dtype_from_np_dtype
from . import constants


class NautilusEstimator(object):

    def __init__(self, model, loss, optimizer, metrics, batch_size, num_epochs):
        """NautilusEstimator Initializer
        Args:
            model (tf.keras.Model): TF Keras Model
            loss (tf.keras.losses.*): TF Keras loss
            optimizer (tf.keras.optimizers.*): TF Keras optimizer
            metrics (list(tf.keras.metrics)): List of TF Keras metrics
            batch_size (int): batch_size
            num_epochs (int): maximum number of epochs
        """
        assert len(model.outputs) == 1, 'Nautilus currently supports only single output models.'
        self.model = model

        self.loss = loss
        self.optimizer = optimizer

        assert type(metrics) == list, 'Metrics has to be a list.'
        assert len(metrics) == 1, 'Nautilus currently supports only single metric.'
        self.metrics = metrics

        self.batch_size = batch_size
        self.num_epochs = num_epochs

    def compile(self):
        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
        return self.model


class NautilusModel(tf.keras.models.Model):
    def __init__(self, *args, **kwargs):
        super(NautilusModel, self).__init__(*args, **kwargs)

    def train_step(self, data):
        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        ys = [y for _ in self.optimizer.optimizers]
        y_preds = self(x, training=True)

        if not self.compiled_loss._built:
            self.compiled_loss.build(y_preds)
        
        for opt, y, y_pred, loss_obj, loss_weight, metric_obj in zip(self.optimizer.optimizers, ys, y_preds, self.compiled_loss._losses,
            self.compiled_loss._loss_weights, self.compiled_loss._per_output_metrics):
            
            loss_value = loss_obj(y, y_pred)
            loss_metric_value = loss_value
            # Correct for the `Mean` loss metrics counting each replica as a batch.
            if loss_obj.reduction == losses_utils.ReductionV2.SUM:
                loss_metric_value *= ds_context.get_strategy().num_replicas_in_sync

            batch_dim = array_ops.shape(y)[0]
            if metric_obj is not None:
                metric_obj.update_state(loss_metric_value, sample_weight=batch_dim)

            if loss_weight is not None:
                loss_value *= loss_weight
                loss_metric_value *= loss_weight

            if (loss_obj.reduction == losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE or
                loss_obj.reduction == losses_utils.ReductionV2.AUTO):
                loss_value = losses_utils.scale_loss_for_distribution(loss_value)

            params = self.trainable_variables
            grads = gradients.gradients(loss_value, params)
            grads = opt._clip_gradients(grads)
            grads_and_vars = [(g, p) for g, p in list(zip(grads, params)) if g is not None]
            opt.apply_gradients(grads_and_vars)

        self.compiled_metrics.update_state(ys, y_preds, sample_weight)
        return {m.name: m.result() for m in self.metrics}


# TODO: Add serializability support
class NautilusOptimizer(tf.keras.optimizers.Optimizer):

    def __init__(self, optimizers, **kwargs):
        self.optimizers = optimizers
        self._opt_index = tf.Variable(-1)

    def get_config(self):
        return {}


def sub_graph_model_gen_fn(model_gen_params, storage_path, custom_objects, memory_budget, return_dict):
    tf.compat.v1.reset_default_graph()
    tf.keras.backend.clear_session()

    with tf.device('/cpu:0'):
        original_models = {}
        for fname in os.listdir(os.path.join(storage_path, 'models')):
            if os.path.isdir(os.path.join(storage_path, 'models', fname)):
                original_models[fname] = tf.keras.models.load_model(os.path.join(
                    storage_path, 'models', fname, '{}.h5'.format(fname)), custom_objects=custom_objects, compile=False)

        for model_name, G,  input_names, output_names in model_gen_params:
            layers = {}
            new_G = nx.DiGraph()
            model_outputs = [get_model_sub_graph_from_materialized_layers(
                G, new_G, output_name, input_names, layers, original_models) for output_name in output_names]
            model_inputs = [layers[input_name] for input_name in input_names if input_name in layers]
            model = tf.keras.Model(inputs=model_inputs, outputs=model_outputs)

            model_path = os.path.join(storage_path, 'models', model_name)
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            model.save(os.path.join(model_path, '{}_modified.h5'.format(model_name)))

            return_dict[model_name] = new_G


def intermediate_features_mat_fn(X_train, X_valid, storage_path, predict_batch_size, memory_budget, custom_objects):
    tf.compat.v1.reset_default_graph()
    tf.keras.backend.clear_session()

    model_path = os.path.join(storage_path, 'models', 'feature_mat', 'feature_mat_modified.h5')
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)

    model_input_layers = [l.name for l in model._input_layers]
    if len(model_input_layers) == 0:
        tf.compat.v1.reset_default_graph()
        return

    model_output_layers = [l.name for l in model._output_layers]
    assert all(x in X_train.keys() for x in model_input_layers)

    Xs = [X_train]
    splits = ['train']
    if X_valid is not None:
        assert all(x in X_valid.keys() for x in model_input_layers)
        Xs.append(X_valid)
        splits.append('valid')

    for X, split in zip(Xs, splits):
        inputs = [X[k] for k in model_input_layers]
        outputs_files = [open(os.path.join(storage_path, 'data', split, '{}.npy'.format(k)), 'ab') for k in model_output_layers]
        num_records = inputs[0].shape[0]
        for i in range(0, num_records, predict_batch_size):
            model_predict_outputs = model.predict([x[i:i+predict_batch_size] for x in inputs], batch_size=predict_batch_size)
            if type(model_predict_outputs) != list:
                model_predict_outputs = [model_predict_outputs]
            num_records = model_predict_outputs[0].shape[0]
            for f, x in zip(outputs_files, model_predict_outputs):
                for i in range(num_records):
                    np.save(f, x[i:i+1])

        for f in outputs_files:
            f.close()

    tf.compat.v1.reset_default_graph()


def model_train_fn(model_name, source_model_names, losses, optimizers, metrics, storage_path, feature_col_names, label_col_names, batch_size, num_epochs,
                   memory_budget, shuffle_buffer_size, return_dict, custom_objects=None):
    tf.compat.v1.reset_default_graph()
    tf.keras.backend.clear_session()

    optimizers = [opt.__class__.from_config(opt.get_config()) for opt in optimizers]

    model_path = os.path.join(storage_path, 'models', model_name, '{}_modified.h5'.format(model_name))
    if not os.path.exists(model_path):
        # No optimized model available. Loading the default model.
        model_path = os.path.join(storage_path, 'models', model_name, '{}.h5'.format(model_name))
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)

    model_input_layers = [l.name for l in model._input_layers]

    def input_iterator(split):
        input_files = [open(os.path.join(storage_path, 'data', split, '{}.npy'.format(l)), 'rb') for l in model_input_layers]
        output_files = [open(os.path.join(storage_path, 'data', split, '{}.npy'.format(l)), 'rb') for l in label_col_names]

        fsz = os.fstat(input_files[0].fileno()).st_size
        while input_files[0].tell() < fsz:
            for input_data, output_data in zip([[np.load(f) for f in input_files]], [[np.load(f) for f in output_files]]):
                if len(label_col_names) == 1:
                    if len(input_data) == 1:
                        yield input_data[0][0], output_data[0][0]
                    else:
                        yield tuple([x[0] for x in input_data]), output_data[0][0]
                else:
                    if len(input_data) == 1:
                        yield input_data[0][0], tuple([y[0] for y in output_data])
                    else:
                        yield tuple([x[0] for x in input_data]), tuple([y[0] for y in output_data])

        for f in input_files + output_files:
            f.close()

    output_types = []
    output_shapes = []
    if len(model.inputs) == 1:
        x = np.load(os.path.join(storage_path, 'data', 'train', '{}.npy'.format(model_input_layers[0])))
        output_types.append(get_tf_dtype_from_np_dtype(x.dtype.name))
        output_shapes.append(x.shape[1:])
    else:
        X = [np.load(os.path.join(storage_path, 'data', 'train', '{}.npy'.format(l))) for l in model_input_layers]
        output_types.append(tuple([get_tf_dtype_from_np_dtype(x.dtype.name) for x in X]))
        output_shapes.append(tuple([x.shape[1:] for x in X]))

    if len(label_col_names) == 1:
        y = np.load(os.path.join(storage_path, 'data', 'train', '{}.npy'.format(label_col_names[0])))
        output_types.append(get_tf_dtype_from_np_dtype(y.dtype.name))
        output_shapes.append(y.shape[1:])
    else:
        Y = [np.load(os.path.join(storage_path, 'data', 'train', '{}.npy'.format(l))) for l in label_col_names]
        output_types.append(tuple([get_tf_dtype_from_np_dtype(y.dtype.name) for y in Y]))
        output_shapes.append(tuple([y.shape[1:] for y in Y]))

    train_dataset = tf.data.Dataset.from_generator(lambda: input_iterator('train'), output_types=tuple(output_types),
                        output_shapes=tuple(output_shapes)).shuffle(shuffle_buffer_size, seed=constants.RANDOM_SEED).batch(batch_size).prefetch(-1)
    valid_dataset = tf.data.Dataset.from_generator(lambda: input_iterator('valid'), output_types=tuple(output_types),
                                                   output_shapes=tuple(output_shapes)).batch(batch_size).prefetch(-1)

    train_history = {}
    for i in range(max(num_epochs)):
        if len(losses) == 1:
            # Single model
            if i == 0:
                # Very first and only compilation
                model.compile(loss=losses[0], optimizer=optimizers[0], metrics=metrics)
            temp_train_history = model.fit(train_dataset, validation_data=valid_dataset, initial_epoch=i, epochs=i+1, verbose=1).history
        else:
            # Multiple fused models
            if i == 0 or any([e == i for e in num_epochs]):
                metrics_map = {}
                losses_map = {}
                nautilus_model_outputs = []
                nautilus_optimizers = []
                for output, output_name, metric, loss, e, opt in zip(model.outputs, model.output_names, metrics, losses, num_epochs, optimizers):
                    if e > i:
                        metrics_map['{}'.format(output_name)] = metric
                        losses_map['{}'.format(output_name)] = loss
                        nautilus_model_outputs.append(output)
                        nautilus_optimizers.append(opt)

                nautilus_model = NautilusModel(inputs=model.inputs, outputs=nautilus_model_outputs)
                nautilus_model.compile(loss=losses_map, optimizer=NautilusOptimizer(nautilus_optimizers), metrics=metrics_map)


            temp_train_history = nautilus_model.fit(train_dataset, validation_data=valid_dataset, initial_epoch=i, epochs=i+1, verbose=1).history
            
        if not train_history:
            train_history = temp_train_history
        else:
            for k in temp_train_history:
                train_history[k].extend(temp_train_history[k])
    
    model_path = os.path.join(storage_path, 'models', model_name, '{}_modified_trained_weights.h5'.format(model_name))
    model.save_weights(model_path)
    tf.compat.v1.reset_default_graph()
    
    if len(source_model_names) > 1:
        # Creating the individual model training histories
        updated_train_history = {k: {} for k in source_model_names}
        for model_name, end_point in zip(source_model_names, model.output_names):
            for k in train_history:
                if k.startswith(end_point):
                    updated_train_history[model_name][k[len(end_point) + 1 : ]] = train_history[k]
                elif k.startswith('val_{}'.format(end_point)):
                    updated_train_history[model_name]['val_' + k[len('val_{}'.format(end_point)) + 1 : ]] = train_history[k]
        train_history = updated_train_history
    else:
        train_history = {source_model_names[0]: train_history}

    return_dict['train_history'] = train_history


def gen_trained_model_with_weights(storage_path, original_model_name, optimized_model_name, loss, metrics, use_optimized, iteration, memory_budget, custom_objects=None):
    tf.compat.v1.reset_default_graph()
    tf.keras.backend.clear_session()

    if use_optimized:
        model_path = os.path.join(storage_path, 'models', original_model_name, '{}.h5'.format(original_model_name))
        original_model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
        
        model_path = os.path.join(storage_path, 'models', optimized_model_name, '{}_modified.h5'.format(optimized_model_name))
        optimized_model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
        
        weights_path = os.path.join(storage_path, 'models', optimized_model_name, '{}_modified_trained_weights.h5'.format(optimized_model_name))
        optimized_model.load_weights(weights_path)
        for l in original_model.layers:
            # Swapping any trainable weights from the trained weights
            if l.trainable and len(l.get_weights()) > 0:
                l.set_weights(optimized_model.get_layer('{}/{}'.format(original_model_name, l.name)).get_weights())
    
    else:
        model_path = os.path.join(storage_path, 'models', optimized_model_name, '{}.h5'.format(optimized_model_name))
        original_model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
        
        weights_path = os.path.join(storage_path, 'models', optimized_model_name, '{}_modified_trained_weights.h5'.format(optimized_model_name))
        original_model.load_weights(weights_path)

    # FIXME: Figureout why this is needed for sparse categorical loss models.
    original_model.compile(loss=loss, metrics=metrics)
    best_models_dir = os.path.join(storage_path, 'models', 'best_models')
    if not os.path.exists(best_models_dir):
        os.makedirs(best_models_dir)
    original_model.save(os.path.join(best_models_dir, '{}.h5'.format(iteration)))
