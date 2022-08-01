# Nautilus

## Introduction
Nautilus is an optimized data system for accelerating deep transfer learning over evolving training datasets. It optimizes the model selection of deep learning models with frozen layers using database inspired techniques. Nautilus is implemented on top TensorFlow and Keras and provides easy to use APIs for defining and executing the deep transfer learning workload. The directories in this repository are organized as follows:

- ./nautilus: Source code for the Nautilus system.
- ./examples: Example end-to-end workloads. These workloads are also used as the expeirmental workloads in the SIGMOD submission.
- ./tests: System test cases.
- ./reproducibility: Experiment scripts, previously run log files, and scripts to reproduce the plots in the SIGMOD submission.


## Trying out the system
### Pre-requisites
Nautilus requires Python 3 (>= 3.5). It also requires the following python libraries:
```
    pip install tensorflow>=2.3.0 # for gpu support use tensorflow-gpu>=2.3.0
    pip install gurobipy>=9.1.2
    pip install tf2onnx
    pip install networkx
    pip install numpy
```

Gurobi (gurobipy) is numerical optimization package. Nautilus's optimizer internally uses Gurobi. You need to [download](https://www.gurobi.com/) Gurobi and set it up to run Nautilus. Gurobi provides a free academic license. You also need to set the `GRB_LICENSE_FILE` environment variable to point to the Gurobi license file.
```
    export GRB_LICENSE_FILE=<gurobi_license_file_path>
```

### Installation
Nautilus can be installed using the following Python command.
```
    python setup.py install
```

### Example Workload

1. First define an estimator generating function. Estimator generating function takes in a parameter value instance and returns an estimator encapsulating a Keras model, which is ready to be trained. E.g.,
```
    import tensorflow as tf
    from tensorflow.keras.layers import Input, Lambda
    from tensorflow.keras.models import Model
    from nautilus import GridSearch, hp_choice, NautilusEstimator, constants, ParamType

    # Custom loss function used by the estimator generating function
    def loss(labels, logits):
        return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

    # Custom accuracy function used by the estimator generating function
    def accuracy(labels, logits):
        return tf.keras.metrics.sparse_categorical_accuracy(labels, logits)

    # Estimator generating function
    def Malaria_ResNet50_estimator_gen_fn(params):
        # Setting random seeds
        tf.random.set_seed(constants.RANDOM_SEED)
        random.seed(constants.RANDOM_SEED)
        np.random.seed(constants.RANDOM_SEED)

        base_model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet',
            input_tensor=Input(shape=(224,224,3), dtype=tf.float32, name='image'), input_shape=(224, 224, 3))
        output = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(base_model.output)
        output = tf.keras.layers.Dense(2, activation='softmax', name='classifier')(output)
        model = tf.keras.models.Model(inputs=base_model.inputs, outputs=output)

        # Freezes some of the layers in this model based on the layer freezing scheme.
        trainable = False
        for l in model.layers:
            l.trainable = trainable
            if l.name == params['frozen_up_to_layer']:
                trainable = True

        optimizer = tf.keras.optimizers.Adam(learning_rate=params['learning_rate'], epsilon=1e-08, clipnorm=1.0)
        return NautilusEstimator(model, loss, optimizer, [accuracy], params['batch_size'], 5)
```

2. Then define a parameter search space. E.g.,
```
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
```

3. Initialize a model selection object such as `GridSearch` or `RandomSearch`. E.g.,
```
    ms = GridSearch(Malaria_ResNet50_estimator_gen_fn, search_space, evaluation_metric='accuracy',
                feature_columns=['image'], label_columns=['label'], custom_objects={'loss':loss, 'accuracy':accuracy})
```

4. Iterative call the `best_model = ms.fit(...)` method by passing new batches of labeled data. For every iteration, the best model for that iteration is returned back. E.g.,
```
    best_model = ms(X_train, y_train, X_valid=X_valid, y_valid=y_valid)
```

More detailed API usage examples can be found in the worklods in the `./examples` directory.