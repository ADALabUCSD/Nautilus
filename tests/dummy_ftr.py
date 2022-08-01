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

def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

def accuracy(labels, logits):
    return tf.keras.metrics.sparse_categorical_accuracy(labels, logits)

def _estimator_gen_fn(params):
    inputs = tf.keras.layers.Input(shape=(1024,))
    x = inputs
    for i in range(params['num_hidden_layers']):
        l = tf.keras.layers.Dense(1024, input_dim=1024, activation='relu', name='layer_{}'.format(i+1))
        l.trainable = False
        x = l(x)
    
    x = tf.keras.layers.Dense(1024, input_dim=1024, activation='softmax', name='classifier')(x)

    model = tf.keras.models.Model(inputs, x)
    optimizer = tf.keras.optimizers.Adam(learning_rate=params['learning_rate'], epsilon=1e-08, clipnorm=1.0)
    return NautilusEstimator(model, loss, optimizer, [accuracy], params['batch_size'], 5)

def test_feature_transfer():
    
    custom_objects = {
        'loss': loss,
        'accuracy': accuracy
    }

    search_space = {
                'num_hidden_layers':  hp_choice(ParamType.ArchitectureTuning, [4, 2]),
                'batch_size': hp_choice(ParamType.HyperparameterTuning, [16, 32]),
                'learning_rate': hp_choice(ParamType.HyperparameterTuning, [5e-5, 5e-6])
            }

    extra_configs = {
        constants.USE_MATERIALIZATION_OPTIMIZATION : not args.no_mat_opt,
        constants.USE_MODEL_MERGE_OPTIMIZATION : not args.no_fuse_opt
    }
    model = GridSearch(_estimator_gen_fn, search_space, evaluation_metric='accuracy',
                    feature_columns=['image'], label_columns=['label'],
                    storage_path=args.storage_path, max_num_records=args.max_num_records, storage_budget=args.storage_budget, memory_budget=args.memory_budget,
                    compute_throughput=args.compute_throughput, disk_throughput=args.disk_throughput, shuffle_buffer_size=args.shuffle_buffer_size,
                    custom_objects=custom_objects, extra_configs=extra_configs)

    assert model is not None

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Argument parser for dummy feature transfer.')
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
    parser.add_argument('--shuffle-buffer-size', help='Number of records in the TensorFlow shuffle buffer size', default=10000, type=float)

    args = parser.parse_args()
    test_feature_transfer()
