import unittest
import warnings
import time
import os
import copy
import argparse
import datetime
import random
import numpy as np
import tensorflow as tf

from commons import NautilusBertEmbeddings, NautilusBertLayer, NautilusBertPooler, NautilusWeightedSum
from commons import load_CoNLL_dataset

from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model
from transformers import TFBertForTokenClassification, BertConfig

from nautilus import constants, ParamType, hp_choice, GridSearch, NautilusEstimator


# Restricting the max runtime memory
memory_limit = os.getenv('NAUTILUS_MEMORY_BUDGET', '10')
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*float(memory_limit))])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)


def loss(labels, logits):
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    active_loss = tf.math.not_equal(tf.reshape(labels, (-1,)), -1)
    reduced_logits = tf.boolean_mask(tf.reshape(logits, (-1, 9)), active_loss)
    labels = tf.boolean_mask(tf.reshape(labels, (-1,)), active_loss)

    return loss_fn(labels, reduced_logits)

def accuracy(labels, logits):
    active_labels_ind = tf.cast(tf.math.not_equal(labels, -1), tf.float32)
    active_labels_count = tf.math.reduce_sum(active_labels_ind, axis=1)
    pred_labels = tf.cast(tf.math.argmax(logits, axis=2), tf.float32)
    matches = tf.math.multiply(active_labels_ind, tf.cast(tf.math.equal(labels, pred_labels), tf.float32))
    matches = tf.math.reduce_sum(matches, axis=1)

    return tf.reshape(tf.math.divide(matches, active_labels_count), (-1, 1))

class FEATURE_TRANSFER_STRATERGY:
    LAST_HIDDEN             = 1
    SECOND_LAST_HIDDEN      = 2
    FOURTH_LAST_HIDDEN      = 3
    SUM_LAST_FOUR_HIDDEN    = 4
    CONCAT_LAST_FOUR_HIDDEN = 5
    EMBEDDINGS              = 6
    SUM_ALL_HIDDEN          = 7


def _CoNLL2_BERT_estimator_gen_fn(params):
    # Setting random seeds
    tf.random.set_seed(constants.RANDOM_SEED)
    random.seed(constants.RANDOM_SEED)
    np.random.seed(constants.RANDOM_SEED)

    if params['no_pretraining']:
        bert_model = TFBertForTokenClassification(BertConfig())
    else:
        bert_model = TFBertForTokenClassification.from_pretrained("bert-base-cased")
    config = bert_model.config
    
    input_ids = Input(shape=(params['input_window_size'],), dtype=tf.int32, name='input_ids')
    attention_mask = Input(shape=(params['input_window_size'],), dtype=tf.int32, name='attention_mask')
    token_type_ids = Input(shape=(params['input_window_size'],), dtype=tf.int32, name='token_type_ids')
    
    layer = NautilusBertEmbeddings(config, bert_model.bert.embeddings, name='embeddings')
    layer.trainable = False
    hidden_states = layer([input_ids, token_type_ids])
    embeddings = hidden_states

    extended_attention_mask = attention_mask[:, tf.newaxis, tf.newaxis, :]
    extended_attention_mask = tf.cast(extended_attention_mask, hidden_states.dtype)
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

    hidden_layer_outputs = []
    for i, layer_module in enumerate(bert_model.bert.encoder.layer):
        layer = NautilusBertLayer(config, layer_module, name='layer_{}'.format(i+1))
        layer.trainable = False
        hidden_states = layer([hidden_states, extended_attention_mask])
        hidden_layer_outputs.append(hidden_states)


    ############### Trainable Layers ##############
    if params['new_layer_type'] == 'bi-lstm':
        new_layer = lambda x: tf.keras.layers.Dense(9, name='classifier')(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(768, return_sequences=True), name='new_layer')(x))
    elif params['new_layer_type'] == 'transformer':
        new_layer = lambda x: tf.keras.layers.Dense(9, name='classifier')(NautilusBertLayer(config, name='new_layer')([x, extended_attention_mask]))
    else:
        new_layer = tf.keras.layers.Dense(9, name='classifier')
        

    if params['feature_transfer_stratergy'] == FEATURE_TRANSFER_STRATERGY.LAST_HIDDEN:
        logits = new_layer(hidden_layer_outputs[-1])
    elif params['feature_transfer_stratergy'] == FEATURE_TRANSFER_STRATERGY.SECOND_LAST_HIDDEN:
        logits = new_layer(hidden_layer_outputs[-2])
    elif params['feature_transfer_stratergy'] == FEATURE_TRANSFER_STRATERGY.EMBEDDINGS:
        logits = new_layer(embeddings)
    elif params['feature_transfer_stratergy'] == FEATURE_TRANSFER_STRATERGY.CONCAT_LAST_FOUR_HIDDEN:
        concat_input = tf.keras.layers.Concatenate(axis=-1)([hidden_layer_outputs[-i] for i in range(1, 5)])
        project_down = tf.keras.layers.Dense(config.hidden_size, name='project_down')(concat_input)
        logits = new_layer(project_down)
    elif params['feature_transfer_stratergy'] == FEATURE_TRANSFER_STRATERGY.SUM_LAST_FOUR_HIDDEN:
        logits = new_layer(NautilusWeightedSum(name='weighted_sum')([hidden_layer_outputs[-i] for i in range(1, 5)]))
    elif params['feature_transfer_stratergy'] == FEATURE_TRANSFER_STRATERGY.SUM_ALL_HIDDEN:
        logits = new_layer(NautilusWeightedSum(name='weighted_sum')([hidden_layer_outputs[i] for i in range(12)]))
    else:
        raise Exception('Unsupported option for fine_tuning_stratergy: {}'.format(params['fine_tuning_stratergy']))

    # CoNLL has 9 labels. We also add PAD (-1).
    if params['new_layer_type'] == 'bi-lstm' and params['fine_tuning_stratergy'] == FEATURE_TRANSFER_STRATERGY.EMBEDDINGS:
        model = Model(inputs=[input_ids, token_type_ids], outputs=logits)
    else:
        model = Model(inputs=[input_ids, attention_mask, token_type_ids], outputs=logits)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=params['learning_rate'], epsilon=1e-08, clipnorm=1.0)
    return NautilusEstimator(model, loss, optimizer, [accuracy], params['batch_size'], params['num_epochs'])


def test_conll_feature_transfer():
    global_begin_time = datetime.datetime.now()
    print('NAUTILUS=>{}: Starting workload'.format(global_begin_time))
    print('NAUTILUS=>{}: RANDOM_SEED: {}'.format(global_begin_time, constants.RANDOM_SEED))
    print('NAUTILUS=>{}: VERY_LARGE_VALUE: {}'.format(global_begin_time, constants.VERY_LARGE_VALUE))
    
    for k in vars(args):
        print('NAUTILUS=>{}: {}: {}'.format(global_begin_time, k.upper(), vars(args)[k]))

    begin_time = datetime.datetime.now()
    print('NAUTILUS=>{}: Starting system initialization'.format(begin_time))

    custom_objects = {
        'NautilusBertEmbeddings': NautilusBertEmbeddings,
        'NautilusBertLayer': NautilusBertLayer,
        'NautilusBertPooler': NautilusBertPooler,
        'NautilusWeightedSum': NautilusWeightedSum,
        'loss': loss,
        'accuracy': accuracy
    }

    if args.mode == 'ftr1':
        batch_sizes = [16, 32]
        learning_rates = [5e-5, 3e-5, 2e-5]
        transfer_layers = [
            FEATURE_TRANSFER_STRATERGY.LAST_HIDDEN,
            FEATURE_TRANSFER_STRATERGY.SECOND_LAST_HIDDEN,
            FEATURE_TRANSFER_STRATERGY.SUM_LAST_FOUR_HIDDEN,
            FEATURE_TRANSFER_STRATERGY.CONCAT_LAST_FOUR_HIDDEN,
            FEATURE_TRANSFER_STRATERGY.EMBEDDINGS,
            FEATURE_TRANSFER_STRATERGY.SUM_ALL_HIDDEN
        ]
        num_epochs = [args.num_epochs]
    elif args.mode == 'ftr2':
        batch_sizes = [16, 32]
        learning_rates = [5e-5, 3e-5, 2e-5]
        transfer_layers = [
            FEATURE_TRANSFER_STRATERGY.LAST_HIDDEN,
            FEATURE_TRANSFER_STRATERGY.SECOND_LAST_HIDDEN,
            FEATURE_TRANSFER_STRATERGY.SUM_LAST_FOUR_HIDDEN,
            FEATURE_TRANSFER_STRATERGY.CONCAT_LAST_FOUR_HIDDEN
        ]
        num_epochs = [args.num_epochs]
    else:
        #workload size exp and ftr3
        transfer_layers = [
            FEATURE_TRANSFER_STRATERGY.CONCAT_LAST_FOUR_HIDDEN
        ]
        if args.mode != 'ftr3':
            # workload size experiment
            batch_sizes = [16]
            num_epochs = [args.num_epochs]
            learning_rates = [1e-4, 5e-5, 3e-5, 2e-5, 1e-5][:int(args.mode.split('-')[1])]
        else:
            # ftr3
            batch_sizes = [16, 32]
            num_epochs = [args.num_epochs, 2*args.num_epochs]
            learning_rates = [5e-5, 3e-5, 2e-5]


    search_space = {
            'feature_transfer_stratergy':  hp_choice(ParamType.ArchitectureTuning, transfer_layers),
            'input_window_size': hp_choice(ParamType.ArchitectureTuning, [128]),
            'new_layer_type': hp_choice(ParamType.ArchitectureTuning, ['transformer']),
            'batch_size': hp_choice(ParamType.HyperparameterTuning, batch_sizes),
            'learning_rate': hp_choice(ParamType.HyperparameterTuning, learning_rates),
            'num_epochs': hp_choice(ParamType.HyperparameterTuning, num_epochs),
            'no_pretraining': hp_choice(ParamType.HyperparameterTuning, [args.no_pretraining])
        }

    extra_configs = {
        constants.USE_MATERIALIZATION_OPTIMIZATION : not args.no_mat_opt,
        constants.USE_MODEL_MERGE_OPTIMIZATION : not args.no_fuse_opt
    }
    model = GridSearch(_CoNLL2_BERT_estimator_gen_fn, search_space, evaluation_metric='accuracy',
                    feature_columns=['input_ids', 'attention_mask', 'token_type_ids'], label_columns=['label'],
                    storage_path=args.storage_path, max_num_records=args.max_num_records, storage_budget=args.storage_budget,
                    memory_budget=args.memory_budget, compute_throughput=args.compute_throughput, disk_throughput=args.disk_throughput, shuffle_buffer_size=args.shuffle_buffer_size,
                    custom_objects=custom_objects, extra_configs=extra_configs)

    assert model is not None
    ct = datetime.datetime.now()
    print('NAUTILUS=>{}: Completed system initialization. Elapsed time: {}'.format(ct, ct - begin_time))        


    for i in range(args.max_num_records//args.active_learning_batch_size):
        ct = datetime.datetime.now()
        print('NAUTILUS=>{}: ========> Active learning cycle: {}'.format(ct, i+1))
        
        begin_time = datetime.datetime.now()
        print('NAUTILUS=>{}: Starting data loading'.format(begin_time))
        X_train, y_train, X_valid, y_valid = load_CoNLL_dataset(max_length=128, batch_size=args.active_learning_batch_size, valid_fraction=args.valid_fraction,
            sampling_stratergy=args.sampling_stratergy, model_path=model.get_path_to_best_model(), custom_objects=custom_objects)
        ct = datetime.datetime.now()
        print('NAUTILUS=>{}: Completed data loading. Elapsed time: {}'.format(ct, ct - begin_time))        
        
        begin_time = datetime.datetime.now()
        print('NAUTILUS=>{}: Starting model selection'.format(begin_time))
        model.fit(X_train, y_train, X_valid=X_valid, y_valid=y_valid, incremental=True)
        ct = datetime.datetime.now()
        print('NAUTILUS=>{}: Completed active learning cycle. Elapsed time: {}'.format(ct, ct - begin_time))        

    ct = datetime.datetime.now()
    print('NAUTILUS=>{}: Workload completed. Elapsed time: {}'.format(ct, ct - global_begin_time))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Argument parser for CoNLL feature transfer.')
    parser.add_argument('--no-mat-opt', help='Whether to disable materialization optimization', default=False, required=False, action='store_true')
    parser.add_argument('--mode', help='Which model selection to run. Options ftr1 (BERT paper) or ftr2 or ftr3', default='ftr1', required=False)
    parser.add_argument('--storage-path', help='Storage directory', default='./storage', required=False)
    parser.add_argument('--no-fuse-opt', help='Whether to disable model fusion optimization', default=False, required=False, action='store_true')
    parser.add_argument('--num-epochs', help='Maximum number of training epchs', default=5, type=int)
    parser.add_argument('--max-num-records', help='Maximum number of records for active learning', default=5000, type=int)
    parser.add_argument('--active-learning-batch-size', help='Size of batch of active learning labeled data', default=500, type=int)
    parser.add_argument('--valid-fraction', help='Fraction of validation data in an active learning batch', default=0.2, type=float)
    parser.add_argument('--sampling-stratergy', help='Active learning sampling stratergy', default='uncertainty')
    parser.add_argument('--no-pretraining', help='Do not use the pre-trained BERT model', default=False, required=False, action='store_true')

    parser.add_argument('--storage-budget', help='Size of storage budget in GBs', default=25, type=float)
    parser.add_argument('--memory-budget', help='Size of memory budget in GBs', default=10, type=float)
    parser.add_argument('--workspace-memory', help='Size of maximum workspace memory in GBs', default=1, type=float)
    parser.add_argument('--compute-throughput', help='Compute throughput of the device in TFLOPs', default=6, type=float)
    parser.add_argument('--disk-throughput', help='Disk read throughput of the device in MB/s', default=500, type=float)
    parser.add_argument('--shuffle-buffer-size', help='Number of records in the TensorFlow shuffle buffer size', default=10000, type=int)

    args = parser.parse_args()
    assert args.mode in ['ftr1', 'ftr2', 'ftr3'] or args.mode.startswith('ftr3-')

    test_conll_feature_transfer()
