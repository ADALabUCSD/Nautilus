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

import numpy as np
import tensorflow as tf
import networkx as nx
from . import constants


def pretty_print(d, indent=0):
    """Pretty prints a nested python dictionary

    Args:
        d (dict): Nested python dictionary
        indent (int, optional): Indentation level. Defaults to 0.
    """
    for key, value in d.items():
        print('\t' * indent + str(key))
        if isinstance(value, dict):
            pretty_print(value, indent+1)
        else:
            print('\t' * (indent+1) + str(value))


def is_valid_evaluation_metric(metric_name):
    """
    Helper method to check whether an evaluating metric is valid/supported.
    :param metric_name:
    :return:
    """
    if metric_name in ['loss', 'acc', 'accuracy']:
        return True
    else:
        return False


def is_lower_evaluation_metric_better(metric_name):
    """
    Helper method to check whether te lower or higher of an eval metric is better.
    :param metric_name:
    :return:
    """
    if metric_name in ['loss']:
        return True
    elif metric_name in  ['acc', 'accuracy']:
        return False
    else:
        raise Exception('Unsupported evaluation metric: {}'.format(metric_name))


def get_dtype_size(dtype):
    if dtype in ['int32', 'float32']:
        return 4
    elif dtype in ['int64', 'float64']:
        return 8
    elif dtype in ['bool']:
        return 1
    else:
        raise Exception('unsupported dtype: {}'.format(dtype))


def get_tf_dtype_from_np_dtype(np_dtype):
    if np_dtype == 'int64':
        return tf.int64
    elif np_dtype == 'int32':
        return tf.int32
    elif np_dtype == 'float64':
        return tf.float64
    elif np_dtype == 'float32':
        return tf.float32
    else:
        raise Exception('Unsupported dtype:{}'.format(np_dtype))


def get_comput_cost_multiply_factor(attrs):
    if attrs['materializable']:
        return 1.0
    elif attrs['frozen']:
        return 2.0
    else:
        return 3.0


def get_storage_size(S, l, max_num_records):
    size = np.prod(S.nodes()[l]['output_shape'][1:]) * get_dtype_size(S.nodes()[l]['layer_config']['dtype']) / (1024.0**2)  # MBs
    return size * max_num_records


def get_load_cost(S, l, max_num_records, materialized_layers, compute_throughput, disk_throughput):
    size = get_storage_size(S, l, max_num_records)
    load_cost = size * 1.0/disk_throughput * compute_throughput if l in materialized_layers else constants.VERY_LARGE_VALUE
    return load_cost


def get_compute_cost(S, l, max_num_records):
    compute_cost = S.nodes()[l]['flops_consumption'] * get_comput_cost_multiply_factor(S.nodes()[l]) * max_num_records
    return compute_cost


def check_layer_configs_match(S, M, layer):
    try:
        np.testing.assert_equal(S.nodes()[layer]['layer_config'], M.nodes()[
                                layer]['layer_config'])
        return True
    except AssertionError as _:
        return False

def get_model_sub_graph_from_materialized_layers(G, new_G, output_name, input_names, layers, original_models=None):
    
    num_epochs = max(G.nodes()[output_name]['num_epochs'])
    
    if output_name in layers:
        assert len(new_G.nodes()[output_name]['num_epochs']) == 1
        nx.set_node_attributes(new_G, {output_name: [max(num_epochs, new_G.nodes()[output_name]['num_epochs'][0])]}, 'num_epochs')
        if original_models is not None:
            return layers[output_name]
        else:
            return None
    elif output_name in input_names:
        node = G.nodes()[output_name]
        attrs = {k: node[k] for k in node if k != 'num_epochs'}
        attrs['num_epochs'] = [num_epochs]
        new_G.add_node(output_name, **attrs)

        if original_models is not None:
            shape = G.nodes()[output_name]['output_shape']
            dtype = G.nodes()[output_name]['layer_config']['dtype']
            layer_input = tf.keras.Input(shape=shape[1:], dtype=dtype, name=output_name)
            layers[output_name] = layer_input
            return layer_input
        else:
            layers[output_name] = None
            return None
    else:
        model = G.nodes()[output_name]['models'][0]
        original_output_name = G.nodes()[output_name]['original_layer_name']
        node = G.nodes()[output_name]
        
        attrs = {k: node[k] for k in node if k != 'num_epochs'}
        attrs['num_epochs'] = [num_epochs]
        new_G.add_node(output_name, **attrs)

        for input_name, _ in G.in_edges(output_name):
            new_G.add_edge(input_name, output_name)

        all_layer_inputs = [get_model_sub_graph_from_materialized_layers(
            G, new_G, input_name, input_names, layers, original_models) for input_name, _ in G.in_edges(output_name)]
        if original_models is not None:
            layer = original_models[model].get_layer(original_output_name)
            layer._name = output_name
            if type(layer.input) == list:
                layers[output_name] = layer(all_layer_inputs)
            else:
                layers[output_name] = layer(all_layer_inputs[0])
            return layers[output_name]
        else:
            layers[output_name] = None
            return None


def merge_model_metadata_graphs(S, M, model_name, num_epochs):
    layer_name_mapping = {}

    def layer_already_exists(S, M, layer, inbound_layers):
        return layer in S.nodes() and S.nodes()[layer]['weights_hash'] == M.nodes()[layer]['weights_hash'] and \
            S.nodes()[layer]['layer_type'] == M.nodes()[layer]['layer_type'] and \
            S.nodes()[layer]['inbound_layers'] == inbound_layers and check_layer_configs_match(S, M, layer)

    for layer in M.nodes():
        materializable = M.nodes()[layer]['materializable']
        inbound_layers = [layer_name_mapping[x] for x, y in list(M.in_edges(layer))]

        if materializable and layer_already_exists(S, M, layer, inbound_layers):
            # A layer match found. Layer matches and all inputs match.
            S.nodes()[layer]['models'].append(model_name)
            S.nodes()[layer]['num_epochs'].append(num_epochs)
            layer_name_mapping[layer] = layer
            continue
        elif (materializable and layer in S.nodes()) or not materializable:
            # Layer by same name exists. But it doesn't match
            # A unique layer name is given.
            # or
            # Layer is not materializable. We treat as a new layer independant of all other models
            # A unique layer name is given.
            layer_name_mapping[layer] = "{}/".format(model_name) + layer
        else:
            # Layer is materializable but has not appeared before
            layer_name_mapping[layer] = layer

        unique_layer_name = layer_name_mapping[layer]
        node = M.nodes()[layer]
        S.add_node(unique_layer_name, **{k: node[k] for k in node})
        for inbound_layer_name in inbound_layers:
            # Add all edges that are not in the multi-model.
            S.add_edge(inbound_layer_name, unique_layer_name)
