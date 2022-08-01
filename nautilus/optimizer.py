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
import copy
import numpy as np
import networkx as nx
from gurobipy import setParam, Model as GurobiModel, GRB
from .utils import get_dtype_size, get_model_sub_graph_from_materialized_layers, get_comput_cost_multiply_factor, get_storage_size, get_load_cost, get_compute_cost
from .profiling import estimate_model_flops_consumption, estimate_model_memory_consumption
from . import constants


def get_materialization_points(S, max_num_records, storage_budget, compute_throughput, disk_throughput):

    output_names = [l for l in S.nodes() if S.out_degree(l) == 0]

    for l in output_names:
        assert len(S.nodes()[l]['num_epochs']) == 1
    num_epochs = [S.nodes()[l]['num_epochs'][0] for l in output_names]

    all_layers = [[o] + [l for l in nx.ancestors(S, o)] for o in output_names]
    all_layers_to_index = [{l: i for i, l in enumerate(layers)} for layers in all_layers]

    materializable_layers = [l for l in S.nodes() if S.nodes()[l]['materializable']]
    materializable_layers_to_index = {l: i for i, l in enumerate(materializable_layers)}

    Ls = [[get_load_cost(S, l, 1, materializable_layers, compute_throughput, disk_throughput)  / constants.VERY_LARGE_VALUE for l in layers] for layers in all_layers]
    Cs = [[get_compute_cost(S, l, 1)  / constants.VERY_LARGE_VALUE for l in layers] for layers in all_layers]

    m = GurobiModel("MILP")

    # 1 => not pruned
    Xas = [m.addVars(range(len(layers)), lb=0, ub=1, vtype=GRB.INTEGER, name='Xa_{}'.format(i)) for i, layers in enumerate(all_layers)]
    # 1 => computed
    Xbs = [m.addVars(range(len(layers)), lb=0, ub=1, vtype=GRB.INTEGER, name='Xb_{}'.format(i)) for i, layers in enumerate(all_layers)]
    # 1 => materialized
    Xcs = m.addVars(range(len(materializable_layers)), lb=0, ub=1, vtype=GRB.INTEGER, name='Xc')


    Z = m.addVar(vtype=GRB.CONTINUOUS, name='Z')

    # Obj. Total evaluation cost
    m.setObjective(Z, GRB.MINIMIZE)
    m.addConstr(Z >= sum([ 
        (Xas[i][j] * Ls[i][j]  + Xbs[i][j] * (Cs[i][j] - Ls[i][j])) * max_num_records * num_epochs[i]
         for i, layers in enumerate(all_layers) for j in range(len(layers))
    ]), 'EVAL_COST')


    # Constraints
    # 1. Outputs has to be computed. Otherwise trivial solution with no computations.
    for i, (output_name, layers_to_index) in enumerate(zip(output_names, all_layers_to_index)):
        m.addConstr((Xas[i][layers_to_index[output_name]] >= 1))

    # 2. Cannot compute and have pruned
    for i, layers in enumerate(all_layers):
        m.addConstrs(((Xas[i][j] - Xbs[i][j] >= 0) for j in range(len(layers))))
    
    # 3. If computed all predecessors has to be not pruned
    for i, layers in enumerate(all_layers):
        for j in range(len(layers)):
            if S.in_degree(layers[j]) > 0:
                in_nodes = [l for l, _ in S.in_edges(layers[j])]
                in_node_indices = [all_layers_to_index[i][l] for l in in_nodes]
                m.addConstr(sum([(Xas[i][k] - Xbs[i][j]) for k in in_node_indices]) >= 0)

    # 4. If loaded has to be materialized
    for i, layers in enumerate(all_layers):
        for j, l in enumerate(layers):
            if l in materializable_layers:
                m.addConstr(Xcs[materializable_layers_to_index[l]] >= Xas[i][j] - Xbs[i][j])

    # 5. Storage constraint (storage budget is in GBs)
    m.addConstr(storage_budget*1024 >= sum([Xcs[materializable_layers_to_index[l]] * get_storage_size(S, l, 1) * max_num_records for l in materializable_layers]), 'STORAGE_COST')

    m.optimize()

    materialized_layers = [l for i, l in enumerate(materializable_layers) if Xcs[i].X > 0.5] # > 0.5 implies 1
    storage_utilization = sum([get_storage_size(S, l, 1) * max_num_records for l in materialized_layers])

    return materialized_layers, storage_utilization


def get_sub_model_and_used_layers(S, materialized_layers, model_outputs, compute_throughput, disk_throughput, max_num_records):
    G = nx.DiGraph()
    layers = {}
    input_names = [n for n in S.nodes() if S.in_degree(n) == 0]
    for output_name in model_outputs:
        assert len(S.nodes()[output_name]['num_epochs']) == 1
        get_model_sub_graph_from_materialized_layers(S, G, output_name, input_names, layers)


    layers = list(G.nodes())
    layers_to_index = {l: i for i, l in enumerate(layers)}
    L = [get_load_cost(S, l, 1, materialized_layers, compute_throughput, disk_throughput)  / constants.VERY_LARGE_VALUE for l in layers]
    C = [get_compute_cost(S, l, 1)  / constants.VERY_LARGE_VALUE for l in layers]

    m = GurobiModel("MILP")
    Xa = m.addVars(range(len(layers)), lb=0, ub=1, vtype=GRB.INTEGER, name='Xa')
    Xb = m.addVars(range(len(layers)), lb=0, ub=1, vtype=GRB.INTEGER, name='Xb')
    Z = m.addVar(vtype=GRB.CONTINUOUS, name='Z')

    for l in layers:
        assert len(G.nodes()[l]['num_epochs']) == 1

    # Obj
    m.setObjective(Z, GRB.MINIMIZE)
    m.addConstr(Z >= sum([(Xa[i] * L[i] + Xb[i] * (C[i] - L[i])) * max_num_records * G.nodes()[layers[i]]['num_epochs'][0] for i in range(len(layers))]), 'EVAL_COST')

    # Constraints
    # 1. Outputs has to be computed. Otherwise trivial solution with no computations.
    for output_name in model_outputs:
        m.addConstr((Xa[layers_to_index[output_name]] >= 1))

    # 2. Cannot compute and have pruned
    m.addConstrs(((Xa[i] - Xb[i] >= 0) for i in range(len(layers))))

    # 3. If computed all predecessors has to be not pruned
    for i in range(len(layers)):
        if G.in_degree(layers[i]) > 0:
            in_nodes = [k for k, _ in G.in_edges(layers[i])]
            in_node_indices = [layers_to_index[k] for k in in_nodes]
            m.addConstr(sum([(Xa[j] - Xb[i]) for j in in_node_indices]) >= 0)

    m.optimize()

    all_used_layers = [layers[i] for i in range(len(layers)) if Xa[i].X > 0.5]
    computed_layers = [layers[i] for i in range(len(layers)) if Xb[i].X > 0.5]

    used_materialized_layers = [l for l in materialized_layers if l in all_used_layers and l not in computed_layers]
    G = nx.DiGraph()
    layers = {}
    for output_name in model_outputs:
        get_model_sub_graph_from_materialized_layers(S, G, output_name, input_names + used_materialized_layers, layers)

    assert all([l in all_used_layers for l in G.nodes()]), 'Inconsistent model graph. Maybe issues with numerical optimization.'
    # Pruning redundant layers that are never used.
    all_used_layers = [l for l in all_used_layers if l in G.nodes()]
    used_materialized_layers = [l for l in used_materialized_layers if l in G.nodes()]
    return G, all_used_layers, used_materialized_layers


def find_execution_plan(S, merged_models, all_materialized_layers, memory_budget, workspace_memory, compute_throughput, disk_throughput, max_num_records):
    merge_candidates = {}
    model_keys = list(merged_models.keys())

    for i in range(len(model_keys)):
        for j in range(i+1, len(model_keys)):
            if merged_models[model_keys[i]]['batch_size'] == merged_models[model_keys[j]]['batch_size']:
                # Important: Combined output layer order is same as 'models' order in merge_candidates.
                # We rely on this order preservation to pass the list of losses and optimizers at training time.
                combined_output_layers = merged_models[model_keys[i]]['output_layers'] + merged_models[model_keys[j]]['output_layers']

                G, all_used, used_materialized = get_sub_model_and_used_layers(S, all_materialized_layers, combined_output_layers, compute_throughput, disk_throughput, max_num_records)

                merge_candidate_models = merged_models[model_keys[i]]['models'] + merged_models[model_keys[j]]['models']
                batch_size = merged_models[model_keys[i]]['batch_size']
                num_epochs = merged_models[model_keys[i]]['num_epochs'] + merged_models[model_keys[j]]['num_epochs']
                mem_consumption = estimate_model_memory_consumption(all_used, S, batch_size, workspace_memory)
                compute_flops, load_flops = estimate_model_flops_consumption(all_used, used_materialized, G, compute_throughput, disk_throughput, max_num_records)
                flops_consumption = compute_flops + load_flops

                flops_reduction = sum([merged_models[model_keys[k]]['flops_consumption'] for k in [i, j]]) - flops_consumption

                if mem_consumption < memory_budget and flops_reduction >= 0:
                    merge_candidates['_'.join([model_keys[i], model_keys[j]])] = {
                        'models': merge_candidate_models,
                        'model_graph': G,
                        'output_layers': combined_output_layers,
                        'flops_consumption': flops_consumption,
                        'compute_flops': compute_flops,
                        'load_flops': load_flops,
                        'memory_consumption': mem_consumption,
                        'used_materialized_layers': used_materialized,
                        'batch_size': batch_size,
                        'num_epochs': num_epochs,
                        'flops_reduction': flops_reduction,
                        'merged_models': [model_keys[i], model_keys[j]]
                    }

    if len(merge_candidates) == 0:
        return merged_models
    else:
        max_key = max(merge_candidates.items(), key=lambda x: x[1]['flops_reduction'])[0]
        best_merged_model = merge_candidates[max_key]
        del merged_models[best_merged_model['merged_models'][0]]
        del merged_models[best_merged_model['merged_models'][1]]
        merged_models[max_key] = copy.copy(best_merged_model)

        return find_execution_plan(S, merged_models, all_materialized_layers, memory_budget, workspace_memory, compute_throughput, disk_throughput, max_num_records)
