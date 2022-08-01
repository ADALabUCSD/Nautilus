import unittest
import warnings
import time
import os
import random
import tensorflow as tf
import numpy as np
import argparse
import datetime
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model
from nautilus import GridSearch, hp_choice, NautilusEstimator, constants, ParamType
from commons import load_malaria_dataset
from nautilus.constants import RANDOM_SEED

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
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

def accuracy(labels, logits):
    return tf.keras.metrics.sparse_categorical_accuracy(labels, logits)

def _Malaria_ResNet50_estimator_gen_fn(params):
    # Setting random seeds
    tf.random.set_seed(constants.RANDOM_SEED)
    random.seed(constants.RANDOM_SEED)
    np.random.seed(constants.RANDOM_SEED)

    base_model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_tensor=Input(shape=(224,224,3), dtype=tf.float32, name='image'), input_shape=(224, 224, 3))
    output = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(base_model.output)
    output = tf.keras.layers.Dense(2, activation='softmax', name='classifier')(output)
    model = tf.keras.models.Model(inputs=base_model.inputs, outputs=output)

    trainable = False
    for l in model.layers:
        l.trainable = trainable
        if l.name == params['frozen_up_to_layer']:
            trainable = True

    optimizer = tf.keras.optimizers.Adam(learning_rate=params['learning_rate'], epsilon=1e-08, clipnorm=1.0)
    return NautilusEstimator(model, loss, optimizer, [accuracy], params['batch_size'], 5)

def test_malaria_fine_tuning():
    global_begin_time = datetime.datetime.now()
    print('NAUTILUS=>{}: Starting workload'.format(global_begin_time))
    print('NAUTILUS=>{}: RANDOM_SEED: {}'.format(global_begin_time, constants.RANDOM_SEED))
    print('NAUTILUS=>{}: VERY_LARGE_VALUE: {}'.format(global_begin_time, constants.VERY_LARGE_VALUE))

    for k in vars(args):
        print('NAUTILUS=>{}: {}: {}'.format(global_begin_time, k.upper(), vars(args)[k]))

    begin_time = datetime.datetime.now()
    print('NAUTILUS=>{}: Starting system initialization'.format(begin_time))

    custom_objects = {
        'loss': loss,
        'accuracy': accuracy
    }

    search_space = {
                'frozen_up_to_layer':  hp_choice(ParamType.HyperparameterTuning, [
                    'conv5_block2_out',
                    'conv5_block1_out',
                    'conv4_block6_out',
                    'conv4_block5_out'
                ]),
                'batch_size': hp_choice(ParamType.HyperparameterTuning, [16, 32]),
                'learning_rate': hp_choice(ParamType.HyperparameterTuning, [5e-5, 3e-5, 2e-5])
            }

    extra_configs = {
        constants.USE_MATERIALIZATION_OPTIMIZATION : not args.no_mat_opt,
        constants.USE_MODEL_MERGE_OPTIMIZATION : not args.no_fuse_opt
    }
    model = GridSearch(_Malaria_ResNet50_estimator_gen_fn, search_space, evaluation_metric='accuracy',
                    feature_columns=['image'], label_columns=['label'],
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
        
        X_train, y_train, X_valid, y_valid = load_malaria_dataset(batch_size=args.active_learning_batch_size, valid_fraction=args.valid_fraction,
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
    test_malaria_fine_tuning()
