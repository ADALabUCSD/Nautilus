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
import tensorflow as tf
from tensorflow.python.util import nest
from tensorflow.python.framework.convert_to_constants import  convert_variables_to_constants_v2_as_graph
from tensorflow.core.framework import types_pb2
import numpy as np
import multiprocessing
import networkx as nx
from tf2onnx.shape_inference import infer_shape_for_graph
from .hparams import ParamType
from . import constants
from .utils import get_dtype_size, get_comput_cost_multiply_factor, get_load_cost, get_compute_cost


def estimate_model_memory_consumption(used_layers, M, batch_size, workspace_memory):
    M = M.subgraph(used_layers)
    M_rev = M.reverse()
    output_nodes = [n for n in M.nodes() if M.out_degree(n) == 0]
    for n in M.nodes():
        if M_rev.nodes()[n]['materializable']:
            M_rev.remove_node(n)

    bp_node_names = sorted(M_rev)
    mapping = {n: 'bp/{}'.format(n) for n in bp_node_names}
    M_rev = nx.relabel_nodes(M_rev, mapping)

    M = nx.compose(M, M_rev)
    # Adding output dependencies
    for n in bp_node_names:
        M.add_edge(n, 'bp/{}'.format(n))

    # Adding input dependencies
    for n in M.nodes():
        if not n.startswith('bp/') and 'bp/{}'.format(n) in M.nodes():
            for i, _ in M.in_edges(n):
                M.add_edge(i, 'bp/{}'.format(n))

    # Adding a (dummy) loss node. Loss node acts as a synchronizer for multi-branch models.
    M.add_node('NAUTILUS/loss', memory_consumption={
        'variable_buffers': 0,
        'trainable_weights': 0,
        'non_trainable_weights': 0,
        'fixed_buffers': 0
    }, output_shape=[-1, 0], layer_config={'dtype': 'int32'}, frozen=False, materializable=False)

    # Adding dependencies to and from loss node
    for on in output_nodes:
        M.add_edge(on, 'NAUTILUS/loss')
        M.add_edge('NAUTILUS/loss', 'bp/{}'.format(on))

    node_order = list(nx.topological_sort(M))
    
    max_memory_utilization = 0
    current_memory_utilization = 0
    node_refs_map = {}

    for n in node_order:
        # Output buffers
        all_intermediate_outputs = M.nodes()[n]['memory_consumption']['variable_buffers'] * batch_size / 1024**3
        final_output = np.prod(M.nodes()[n]['output_shape'][1:]) * get_dtype_size(M.nodes()[n]['layer_config']['dtype']) / 1024.0**3

        # Following is a fix to reduce the highly bloated layer-wise memory utilization in the presence of composite layers.
        # If the layer is materializable we consider only the final output and consider the sum of all intermediate buffers for the
        # calculation of max workspace memory.
        if M.nodes()[n]['materializable']:
            # Materializable layer. Storing all intermediate outputs until completion is not needed.
            outputs = final_output
        else:
            outputs = all_intermediate_outputs

        current_memory_utilization += outputs
        max_memory_utilization = max(max_memory_utilization, current_memory_utilization)

        node_refs_map[n] = {'num_refs': M.out_degree(n), 'mem_footprint': outputs}
        for i, _ in M.in_edges(n):
            if node_refs_map[i]['num_refs'] == 1:
                # Node going out of scope
                current_memory_utilization -= node_refs_map[i]['mem_footprint']
            node_refs_map[i]['num_refs'] -= 1

    weights = sum([M.nodes()[l]['memory_consumption']['fixed_buffers'] / 1024**3 for l in used_layers])
    
    # Inter operator parallelism for workspace memory.
    # https://stackoverflow.com/questions/41233635/meaning-of-inter-op-parallelism-threads-and-intra-op-parallelism-threads
    if len(tf.config.list_physical_devices('GPU')) == 0:
        # Running on CPU. Have to account for interop parallelism (TF GPU runs one op at a time).
        if tf.config.threading.get_inter_op_parallelism_threads() <= 0:
            workspace_memory *= multiprocessing.cpu_count()
        else:
            workspace_memory *= tf.config.threading.get_inter_op_parallelism_threads()

    max_memory_utilization = weights + workspace_memory + max_memory_utilization
   
    return max_memory_utilization


def estimate_model_flops_consumption(used_layers, used_materialized_layers, M, compute_throughput, disk_throughput, max_num_records):
    input_layers = [l for l in M.nodes() if M.in_degree(l) == 0]
    computed_layers = [l for l in used_layers if l not in used_materialized_layers + input_layers]
    loaded_layers = [l for l in used_layers if l in used_materialized_layers + input_layers]

    for l in computed_layers + loaded_layers:
        assert len(M.nodes()[l]['num_epochs']) == 1

    compute_cost = sum([get_compute_cost(M, l, 1) * max_num_records * M.nodes()[l]['num_epochs'][0] for l in computed_layers])
    load_cost = sum([get_load_cost(M, l, 1, loaded_layers, compute_throughput, disk_throughput) * max_num_records * M.nodes()[l]['num_epochs'][0] for l in loaded_layers])
    return compute_cost, load_cost


def get_memory_consumption_by_layer(model, graph_def, batch_size=1):
    layer_mem_summaries = {layer.name:
                            {
                               'non_trainable_weights': sum([tf.keras.backend.count_params(p) * p.dtype.size for p in layer.non_trainable_weights]),
                               'trainable_weights': sum([tf.keras.backend.count_params(p) * p.dtype.size for p in layer.trainable_weights]),
                               'fixed_buffers': 0,
                               'variable_buffers': 0
                            } for layer in model.layers}

    for node in graph_def.node:
        all_shapes = []

        for shape in node.attr['_output_shapes'].list.shape:
            all_shapes.append([d.size for d in shape.dim])

        layer_name = node.name
        if layer_name.startswith('functional_1'):
            # Has the form 'functional_1/<layer_name>/...'
            layer_name = layer_name.split('/')[1]

        if len(all_shapes) >= 1 and layer_name in layer_mem_summaries:
            
            for shapes in all_shapes:
                dtype_size = -1
                if node.attr['dtype'].type == types_pb2.DT_FLOAT or node.attr['T'].type == types_pb2.DT_FLOAT:
                    # float32
                    dtype_size = 4
                elif node.attr['dtype'].type == types_pb2.DT_DOUBLE or node.attr['T'].type == types_pb2.DT_DOUBLE:
                    # float64
                    dtype_size = 8
                elif node.attr['dtype'].type == types_pb2.DT_INT32 or node.attr['T'].type == types_pb2.DT_INT32:
                    # int32
                    dtype_size = 4
                elif node.attr['dtype'].type == types_pb2.DT_UINT8 or node.attr['T'].type == types_pb2.DT_UINT8:
                    # uint8
                    dtype_size = 1
                elif node.attr['dtype'].type == types_pb2.DT_INT16 or node.attr['T'].type == types_pb2.DT_INT16:
                    # int16
                    dtype_size = 2
                elif node.attr['dtype'].type == types_pb2.DT_INT8 or node.attr['T'].type == types_pb2.DT_INT8:
                    # int8
                    dtype_size = 1
                elif node.attr['dtype'].type == types_pb2.DT_INT64 or node.attr['T'].type == types_pb2.DT_INT64:
                    # int64
                    dtype_size = 8
                elif node.attr['dtype'].type == types_pb2.DT_BOOL or node.attr['T'].type == types_pb2.DT_BOOL:
                    # bool
                    dtype_size = 1/8.
                elif node.attr['dtype'].type == types_pb2.DT_UINT16 or node.attr['T'].type == types_pb2.DT_UINT16:
                    # uint16
                    dtype_size = 2
                elif node.attr['dtype'].type == types_pb2.DT_HALF or node.attr['T'].type == types_pb2.DT_HALF:
                    # float16
                    dtype_size = 2
                elif node.attr['dtype'].type == types_pb2.DT_UINT32 or node.attr['T'].type == types_pb2.DT_UINT32:
                    # uint32
                    dtype_size = 4
                elif node.attr['dtype'].type == types_pb2.DT_UINT64 or node.attr['T'].type == types_pb2.DT_UINT64:
                    # uint64
                    dtype_size = 8
                elif node.attr['dtype'].type == types_pb2.DT_INVALID or node.attr['T'].type == types_pb2.DT_INVALID:
                    # Cannot find dtype. Assumes int32 or float32
                    dtype_size = 4
                else:
                    raise Exception('Unsupported dtype: {}'.format(node))
                
                if -1 in shapes:
                    layer_mem_summaries[layer_name]['variable_buffers'] += np.prod([s if s != -1 else batch_size for s in shapes]) * dtype_size
                else:
                    layer_mem_summaries[layer_name]['fixed_buffers'] += np.prod(shapes) * dtype_size

    return layer_mem_summaries


def get_flops_consumption_by_layer(model, graph_def):
    with tf.Graph().as_default() as graph:
        tf.graph_util.import_graph_def(graph_def, name='')
        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd="scope", options=opts)

    layer_flop_summaries = {l.name: 0 for l in model.layers}

    for g in flops.children:
        if hasattr(g, 'total_float_ops'):
            # E.g., format: "functional_1/dense/BiasAdd"
            layer_name = g.name.split('/')[1]
            if layer_name in layer_flop_summaries:
                layer_flop_summaries[layer_name] += g.total_float_ops

    return layer_flop_summaries


def generate_metadata_graph(model_name, model, num_epochs):
    M = nx.DiGraph()
    # Traverse the nodes in the graph in topologically sorted order.
    for layer in model.layers:

        # A layer is frozen if it is not trainable and has empty weights
        frozen = not layer.trainable or len(layer.get_weights()) == 0
        materializable = frozen

        inbound_layers = []
        for i, node in enumerate(layer._inbound_nodes):
            node_key = layer.name + '_ib-' + str(i)
            if node_key in model._network_nodes:
                for inbound_layer in nest.flatten(node.inbound_layers):
                    inbound_layers.append(inbound_layer.name)
                    # A layer is not materializable only if it is frozen and all its inputs are materializable
                    materializable = materializable and nx.get_node_attributes(M, "materializable")[inbound_layer.name]

        if type(layer.output_shape) == list:
            assert len(layer.output_shape) == 1, 'Currently support only single output layers'

        M.add_node(
            layer.name, frozen=frozen, materializable=materializable,
            original_layer_name=layer.name, models=[model_name], num_epochs=[num_epochs],
            output_shape=layer.output_shape if type(layer.output_shape) == tuple else layer.output_shape[0],
            weights_hash=sum([hash(x.data.tobytes()) for x in layer.get_weights()]),
            layer_type=type(layer),
            layer_config=layer.get_config(),
            inbound_layers=inbound_layers
        )

        for inbound_layer_name in inbound_layers:
            M.add_edge(inbound_layer_name, layer.name)

    return M


def model_initialization_fn(model_name, storage_path, estimator_gen_fn, params, param_types, benchmark, custom_objects, return_dict):
    tf.compat.v1.reset_default_graph()
    
    with tf.device('/cpu:0'):
        # creating model
        estimator = estimator_gen_fn(params)
        if not os.path.exists(os.path.join(storage_path, 'models', '{}'.format(model_name))):
            os.makedirs(os.path.join(storage_path, 'models', '{}'.format(model_name)))

        # saving model
        model = estimator.compile()
        model.save(os.path.join(storage_path, 'models', '{}'.format(model_name), '{}.h5'.format(model_name)))

        M = generate_metadata_graph(model_name, model, estimator.num_epochs)
        architecture_params = [k for k in param_types if param_types[k] == ParamType.ArchitectureTuning]
        architecture_param_vals = [params[k] for k in architecture_params]
        arch_dep_params_unique_key = "/".join([str(v) for v in architecture_param_vals])

        # benchmarking the model
        if benchmark:
            found = False
            for k in return_dict.keys():
                if return_dict[k]['archi_dep_params_key'] == arch_dep_params_unique_key:
                    found = True
                    # Found the profiling information for a structurally similar model
                    prev_M = return_dict[k]['metadata_graph']
                    nx.set_node_attributes(M, nx.get_node_attributes(prev_M,'memory_consumption'), "memory_consumption")
                    nx.set_node_attributes(M, nx.get_node_attributes(prev_M,'flops_consumption'), "flops_consumption")
                    break
            
            if not found:
                exec("def model_func({}): model([{}])".format(', '.join([i.name.split(':')[0] for i in model.inputs]), ', '.join([i.name.split(':')[0] \
                     for i in model.inputs])), locals(), locals())
                concrete = tf.function(func=locals()['model_func'])
                concrete_func = concrete.get_concrete_function(**{i.name.split(':')[0]:tf.TensorSpec([None, *i.shape[1:]]) for i in model.inputs})
                memory_consumption_by_layer = get_memory_consumption_by_layer(model, infer_shape_for_graph(concrete_func.graph).as_graph_def(add_shapes=True), batch_size=1)

                concrete_func = concrete.get_concrete_function(**{i.name.split(':')[0]:tf.TensorSpec([1, *i.shape[1:]]) for i in model.inputs})
                _, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)
                flops_consumption_by_layer = get_flops_consumption_by_layer(model, graph_def)
                
                nx.set_node_attributes(M, memory_consumption_by_layer, "memory_consumption")
                nx.set_node_attributes(M, flops_consumption_by_layer, "flops_consumption")

        return_dict[model_name] = {
            'batch_size': estimator.batch_size,
            'num_epochs': estimator.num_epochs,
            'loss': estimator.loss,
            'optimizer': estimator.optimizer,
            'metrics': estimator.metrics,
            'metadata_graph': M,
            'archi_dep_params_key': arch_dep_params_unique_key
        }

