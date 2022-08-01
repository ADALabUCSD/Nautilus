import unittest
import warnings
import time
import os
import tensorflow as tf
import argparse
import random
import numpy as np
import datetime
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model
from transformers import TFBertForSequenceClassification, TFBertForTokenClassification

from nautilus import GridSearch, hp_choice, NautilusEstimator, constants, ParamType

from commons import (NautilusBertEmbeddings, NautilusBertLayer, NautilusBertPooler, NautilusBertAdapter,
    NautilusBertMultiHeadAttention, NautilusBertAddLayerNorm, NautilusBertFeedForward, NautilusBertIntermediate)
from commons import load_CoNLL_dataset

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


def _CoNLL_BERT_estimator_gen_fn(params):
    # Setting random seeds
    tf.random.set_seed(constants.RANDOM_SEED)
    random.seed(constants.RANDOM_SEED)
    np.random.seed(constants.RANDOM_SEED)

    def get_output(inputs, bert_layer, config, name_prefix, add_adapter=False):
        mha = NautilusBertMultiHeadAttention(config, bert_layer.attention.self_attention, name=name_prefix+"mha")
        mha.trainable = False
        
        ff1 = NautilusBertFeedForward(config, bert_layer.attention.dense_output.dense, name=name_prefix+"ff1")
        ff1.trainable = False

        addnorm1 = NautilusBertAddLayerNorm(config, bert_layer.attention.dense_output.LayerNorm, name=name_prefix+"ln1")
        addnorm1.trainable = add_adapter

        intermediate = NautilusBertIntermediate(config, bert_layer.intermediate, name=name_prefix+"intermediate")
        intermediate.trainable = False

        ff2 = NautilusBertFeedForward(config, bert_layer.bert_output.dense, name=name_prefix+"ff2")
        ff2.trainable = False

        addnorm2 = NautilusBertAddLayerNorm(config, bert_layer.bert_output.LayerNorm, name=name_prefix+"ln2")
        addnorm2.trainable = add_adapter

        hidden_states = mha([inputs[0], inputs[1]])
        hidden_states = ff1(hidden_states)
        if add_adapter:
            ada1 = NautilusBertAdapter(config, name=name_prefix+"ada1")
            ada1.trainable = True
            hidden_states = ada1(hidden_states)
        hidden_states = addnorm1([hidden_states, inputs[0]])
        
        residual = hidden_states
        hidden_states = intermediate(hidden_states)
        hidden_states = ff2(hidden_states)
        if add_adapter:
            ada2 = NautilusBertAdapter(config, name=name_prefix+"ada2")
            ada2.trainable = True
            hidden_states = ada2(hidden_states)
        hidden_states = addnorm2([hidden_states, residual])

        return hidden_states


    bert_model = TFBertForTokenClassification.from_pretrained("bert-base-cased")
    config = bert_model.config
    
    input_ids = Input(shape=(params['input_window_size'],), dtype=tf.int32, name='input_ids')
    attention_mask = Input(shape=(params['input_window_size'],), dtype=tf.int32, name='attention_mask')
    token_type_ids = Input(shape=(params['input_window_size'],), dtype=tf.int32, name='token_type_ids')
    
    layer = NautilusBertEmbeddings(config, bert_model.bert.embeddings, name='embeddings')
    layer.trainable = False
    hidden_states = layer([input_ids, token_type_ids])

    extended_attention_mask = attention_mask[:, tf.newaxis, tf.newaxis, :]
    extended_attention_mask = tf.cast(extended_attention_mask, hidden_states.dtype)
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

    hidden_layer_outputs = []
    for i, layer_module in enumerate(bert_model.bert.encoder.layer):
        hidden_states = get_output([hidden_states, extended_attention_mask], layer_module, config, 'layer_{}_'.format(i+1), add_adapter=i>=params['adapters_from']-1)
        hidden_layer_outputs.append(hidden_states)

    new_layer = tf.keras.layers.Dense(9, name='classifier')
    logits = new_layer(hidden_layer_outputs[-1])
    model = Model(inputs=[input_ids, attention_mask, token_type_ids], outputs=logits)

    optimizer = tf.keras.optimizers.Adam(learning_rate=params['learning_rate'], epsilon=1e-08, clipnorm=1.0)
    return NautilusEstimator(model, loss, optimizer, [accuracy], params['batch_size'], 5)


def test_conll_adapter_training():
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
        'NautilusBertAdapter': NautilusBertAdapter,
        'NautilusBertMultiHeadAttention': NautilusBertMultiHeadAttention,
        'NautilusBertAddLayerNorm': NautilusBertAddLayerNorm,
        'NautilusBertFeedForward': NautilusBertFeedForward,
        'NautilusBertIntermediate': NautilusBertIntermediate,
        'loss': loss,
        'accuracy': accuracy
    }
    search_space = {
                'adapters_from':  hp_choice(ParamType.ArchitectureTuning, [9, 10, 11, 12]),
                'input_window_size': hp_choice(ParamType.ArchitectureTuning, [128]),
                'batch_size': hp_choice(ParamType.HyperparameterTuning, [16, 32]),
                'learning_rate': hp_choice(ParamType.HyperparameterTuning, [5e-5, 3e-5, 2e-5])
            }

    extra_configs = {
        constants.USE_MATERIALIZATION_OPTIMIZATION : not args.no_mat_opt,
        constants.USE_MODEL_MERGE_OPTIMIZATION : not args.no_fuse_opt
    }
    model = GridSearch(_CoNLL_BERT_estimator_gen_fn, search_space, evaluation_metric='accuracy',
                    feature_columns=['input_ids', 'attention_mask', 'token_type_ids'], label_columns=['label'],
                    storage_path=args.storage_path, max_num_records=args.max_num_records, storage_budget=args.storage_budget, memory_budget=args.memory_budget,
                    compute_throughput=args.compute_throughput, disk_throughput=args.disk_throughput, shuffle_buffer_size=args.shuffle_buffer_size,
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
    parser.add_argument('--no-fuse-opt', help='Whether to disable model fusion optimization', default=False, required=False, action='store_true')
    parser.add_argument('--storage-path', help='Storage directory', default='./storage', required=False)
    parser.add_argument('--max-num-records', help='Maximum number of records for active learning', default=5000, type=int)
    parser.add_argument('--active-learning-batch-size', help='Size of batch of active learning labeled data', default=500, type=int)
    parser.add_argument('--valid-fraction', help='Fraction of validation data in an active learning batch', default=0.2, type=float)
    parser.add_argument('--sampling-stratergy', help='Active learning sampling stratergy', default='uncertainty')

    parser.add_argument('--storage-budget', help='Size of storage budget in GBs', default=25, type=float)
    parser.add_argument('--memory-budget', help='Size of memory budget in GBs', default=10, type=float)
    parser.add_argument('--workspace-memory', help='Size of maximum workspace memory in GBs', default=1, type=float)
    parser.add_argument('--compute-throughput', help='Compute throughput of the device in TFLOPs', default=6, type=float)
    parser.add_argument('--disk-throughput', help='Disk read throughput of the device in MB/s', default=500, type=float)
    parser.add_argument('--shuffle-buffer-size', help='Number of records in the TensorFlow shuffle buffer size', default=10000, type=int)
    
    args = parser.parse_args()
    test_conll_adapter_training()
