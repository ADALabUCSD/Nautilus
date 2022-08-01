import random
import multiprocessing
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nautilus import constants
from datasets import load_dataset
from transformers import (TFBertForSequenceClassification, BertTokenizer, TFBertMainLayer)
from transformers.models.bert.modeling_tf_bert import (TFBertEmbeddings, TFBertLayer, TFBertPooler, TFBertSelfAttention, TFBertIntermediate)

from nautilus.constants import RANDOM_SEED

# Setting random seeds
tf.random.set_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


DATASETS = {}

class NautilusBertAdapter(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        if isinstance(config, dict):
            from transformers import PretrainedConfig
            config = PretrainedConfig.from_dict(config)
        self.config = config

        self.adapter_down = tf.keras.layers.Dense(units=64, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=1e-3), activation=tf.keras.activations.elu)
        self.adapter_up = tf.keras.layers.Dense(units=config.hidden_size, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=1e-3))

    def call(self, input):
        return self.adapter_up(self.adapter_down(input))
    
    def get_config(self):
        cfg = super().get_config()
        cfg['config'] = self.config.to_dict()
        return cfg


class NautilusBertFeedForward(tf.keras.layers.Layer):
    def __init__(self, config, dense=None, **kwargs):
        super().__init__(**kwargs)
        if isinstance(config, dict):
            from transformers import PretrainedConfig
            config = PretrainedConfig.from_dict(config)
        self.config = config

        if dense is not None:
            self.dense = dense
        else:
            self.dense =  tf.keras.layers.Dense(config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense")

    def call(self, input):
        return self.dense(input)

    def get_config(self):
        cfg = super().get_config()
        cfg['config'] = self.config.to_dict()
        return cfg


class NautilusBertIntermediate(tf.keras.layers.Layer):
    def __init__(self, config, intermediate=None, **kwargs):
        super().__init__(**kwargs)
        if isinstance(config, dict):
            from transformers import PretrainedConfig
            config = PretrainedConfig.from_dict(config)
        self.config = config

        if intermediate is not None:
            self.intermediate = intermediate
        else:
            self.intermediate =  TFBertIntermediate(config)

    def call(self, input):
        return self.intermediate(input)

    def get_config(self):
        cfg = super().get_config()
        cfg['config'] = self.config.to_dict()
        return cfg


class NautilusBertAddLayerNorm(tf.keras.layers.Layer):
    def __init__(self, config, layer_norm=None, **kwargs):
        super().__init__(**kwargs)
        if isinstance(config, dict):
            from transformers import PretrainedConfig
            config = PretrainedConfig.from_dict(config)
        self.config = config

        if layer_norm is not None:
            self.layer_norm = layer_norm
        else:
            self.layer_norm =  tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps)

    def call(self, inputs):
        return self.layer_norm(inputs[0] + inputs[1])

    def get_config(self):
        cfg = super().get_config()
        cfg['config'] = self.config.to_dict()
        return cfg


class NautilusBertMultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, config, self_attention=None, **kwargs):
        super().__init__(**kwargs)
        if isinstance(config, dict):
            from transformers import PretrainedConfig
            config = PretrainedConfig.from_dict(config)
        self.config = config

        if self_attention is not None:
            self.self_attention = self_attention
        else:
            self.self_attention =  TFBertSelfAttention(config)

    def call(self, inputs):
        output = self.self_attention(inputs[0], inputs[1], None, None, training=False)
        return tf.reshape(output[0], tf.shape(inputs[0]))

    def get_config(self):
        cfg = super().get_config()
        cfg['config'] = self.config.to_dict()
        return cfg


class NautilusBertEmbeddings(tf.keras.layers.Layer):
    def __init__(self, config, hf_bert_embeddings=None, **kwargs):
        super().__init__(**kwargs)
        if isinstance(config, dict):
            from transformers import PretrainedConfig
            config = PretrainedConfig.from_dict(config)
        self.config = config
        if hf_bert_embeddings is not None:
            self.embeddings = hf_bert_embeddings
        else:
            self.embeddings = TFBertEmbeddings(config)
        
    def call(self, inputs):
        return self.embeddings(inputs[0], None, inputs[1], None, training=False)
    
    def get_config(self):
        cfg = super().get_config()
        cfg['config'] = self.config.to_dict()
        return cfg


class NautilusBertLayer(tf.keras.layers.Layer):
    def __init__(self, config, hf_bert_layer=None, **kwargs):
        super().__init__(**kwargs)
        if isinstance(config, dict):
            from transformers import PretrainedConfig
            config = PretrainedConfig.from_dict(config)
        self.config = config
        if hf_bert_layer is not None:
            self.hf_bert_layer = hf_bert_layer
        else:
            self.hf_bert_layer = TFBertLayer(config)

    def call(self, inputs):
        return self.hf_bert_layer(inputs[0], inputs[1], None, None, training=False)[0]
    
    def get_config(self):
        cfg = super().get_config()
        cfg['config'] = self.config.to_dict()
        return cfg


class NautilusBertPooler(tf.keras.layers.Layer):
    def __init__(self, config, hf_bert_pooler=None, **kwargs):
        super().__init__(**kwargs)
        if isinstance(config, dict):
            from transformers import PretrainedConfig
            config = PretrainedConfig.from_dict(config)
        self.config = config
        if hf_bert_pooler is not None:
            self.hf_bert_pooler = hf_bert_pooler
        else:
            self.hf_bert_pooler = TFBertPooler(config)
        
    def call(self, inputs):
        return self.hf_bert_pooler.dense(tf.gather(inputs, tf.constant([0]), axis=1))
    
    def get_config(self):
        cfg = super().get_config()
        cfg['config'] = self.config.to_dict()
        return cfg


class NautilusWeightedSum(tf.keras.layers.Layer):
    """A custom keras layer to learn a weighted sum of tensors"""

    def __init__(self, **kwargs):
        super(NautilusWeightedSum, self).__init__(**kwargs)

    def build(self, input_shape=1):
        self.alphas = [self.add_weight(
            name='alpha_{}'.format(i),
            shape=(),
            initializer='ones',
            dtype='float32',
            trainable=True,
        ) for i in range(len(input_shape))]
        super(NautilusWeightedSum, self).build(input_shape)

    def call(self, model_outputs):
        return tf.keras.layers.add([self.alphas[i] * model_outputs[i] for i in range(len(model_outputs))])

    def compute_output_shape(self, input_shape):
        return input_shape[0]


def get_initializer(initializer_range = 0.02):
    return tf.keras.initializers.TruncatedNormal(stddev=initializer_range)


def conll_uncertainty_sampling(X, model_path, return_dict, custom_objects):
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
    y = model.predict(X)
    probs = np.exp(y)/np.sum(np.exp(y), axis=-1, keepdims=True)
    lc = (1 - np.max(probs, axis=-1)) * probs.shape[-1] / (probs.shape[-1] - 1)
    attention_mask = X['attention_mask']
    return_dict['result'] = np.sum((lc * attention_mask), axis=-1) / np.sum(attention_mask, axis=-1)


def malaria_uncertainty_sampling(X, model_path, return_dict, custom_objects):
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
    y = model.predict(X)
    probs = np.exp(y)/np.sum(np.exp(y), axis=-1, keepdims=True)
    lc = (1 - np.max(probs, axis=-1)) * probs.shape[-1] / (probs.shape[-1] - 1)
    return_dict['result'] = lc


def sample_data(dataset_name, X, y, batch_size, sampling_stratergy, model_path, custom_objects={}):
    assert len(y['label']) >= batch_size, 'Exhausted training data.'

    if sampling_stratergy=='random' or model_path is None:
        X_train_batch = {k : X[k][:batch_size] for k in X}
        y_train_batch = {k : y[k][:batch_size] for k in y}

        DATASETS[dataset_name]['X_train'] = {k : X[k][batch_size:] for k in X}
        DATASETS[dataset_name]['y_train'] = {k : y[k][batch_size:] for k in y}

        return X_train_batch, y_train_batch

    elif sampling_stratergy=='uncertainty':
        import threading
        import queue

        if dataset_name == 'conll2003':
            uncertainty_sampling = conll_uncertainty_sampling
        elif dataset_name == 'malaria':
            uncertainty_sampling = malaria_uncertainty_sampling
        else:
            raise Exception('Unsupported dataset: {}'.format(dataset_name))

        manager = multiprocessing.Manager()
        return_dict = manager.dict()

        p = multiprocessing.Process(target=uncertainty_sampling, args=(X, model_path, return_dict, custom_objects))
        p.start()
        p.join()
        result = return_dict['result']

        sorted_result = result.argsort()
        chosen_indices = sorted_result[-batch_size:][::-1]
        not_chosen_indices = sorted_result[:-batch_size][::-1]

        DATASETS[dataset_name]['X_train'] = {k : X[k][not_chosen_indices] for k in X}
        DATASETS[dataset_name]['y_train'] = {k : y[k][not_chosen_indices] for k in y}

        X_train_batch = {k : X[k][chosen_indices] for k in X}
        y_train_batch = {k : y[k][chosen_indices] for k in y}

        return X_train_batch, y_train_batch
    else:
        raise Exception('Unsupported sampling stratergy: {}'.format(sampling_stratergy))


def load_malaria_dataset(max_records_to_load=8000, batch_size=500, valid_fraction=0.2, sampling_stratergy='random', model_path=None, custom_objects={}):
    if 'malaria' not in DATASETS:
        ds = tfds.load('malaria', split='train', as_supervised=True)
        ds = ds.map(lambda x,y: (tf.image.resize(x, [224, 224]), y))
        X = []
        y = []

        for i, (ax, ay) in enumerate(tfds.as_numpy(ds)):
            # Enforcing a max number of records
            if i >= max_records_to_load:
                break
            X.append(ax)
            y.append(ay)

        X_train = {'image': np.stack(X[:int((1-valid_fraction)*max_records_to_load)], axis=0)}
        y_train = {'label': np.array(y[:int((1-valid_fraction)*max_records_to_load)])}

        X_valid = {'image': np.stack(X[int((1-valid_fraction)*max_records_to_load):max_records_to_load], axis=0)}
        y_valid = {'label': np.array(y[int((1-valid_fraction)*max_records_to_load):max_records_to_load])}

        DATASETS['malaria'] = {
                'X_train' : X_train,
                'y_train' : y_train,
                'X_valid' : X_valid,
                'y_valid' : y_valid
            }
    else:
        X_train = DATASETS['malaria']['X_train']
        y_train = DATASETS['malaria']['y_train']
        X_valid = DATASETS['malaria']['X_valid']
        y_valid = DATASETS['malaria']['y_valid']


    X_train_batch, y_train_batch = sample_data('malaria', X_train, y_train, int((1-valid_fraction) * batch_size), sampling_stratergy, model_path, custom_objects)

    validation_size = int(valid_fraction * batch_size)
    X_valid_batch = {k: X_valid[k][:validation_size] for k in X_valid}
    y_valid_batch = {k: y_valid[k][:validation_size] for k in y_valid}

    DATASETS['malaria']['X_valid'] = {k: X_valid[k][validation_size:] for k in X_valid}
    DATASETS['malaria']['y_valid'] = {k: y_valid[k][validation_size:] for k in y_valid}

    return  X_train_batch, y_train_batch, X_valid_batch, y_valid_batch


def load_CoNLL_dataset(max_records_to_load=10000, max_length=128, batch_size=500, valid_fraction=0.2, sampling_stratergy='random', model_path=None, custom_objects={}):
    if 'conll2003' not in DATASETS:
        data = load_dataset('conll2003').shuffle(seed=constants.RANDOM_SEED)
        train_dataset = data["train"]

        train_sentences = [x['tokens'] for x in train_dataset]
        train_labels = [x['ner_tags'] for x in train_dataset]

        tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)

        def tokenize_and_preserve_labels(sentence, text_labels):
            tokenized_sentence = []
            labels = []

            for word, label in zip(sentence, text_labels):

                # Tokenize the word and count # of subwords the word is broken into
                tokenized_word = tokenizer.tokenize(word)
                n_subwords = len(tokenized_word)

                # Add the tokenized word to the final tokenized word list
                tokenized_sentence.extend(tokenized_word)

                # Add the same label to the new list of labels `n_subwords` times
                labels.extend([label] * n_subwords)

            return tokenized_sentence, labels

        train_tokenized_texts_and_labels = [
            tokenize_and_preserve_labels(sent, labs)
            for sent, labs in zip(train_sentences, train_labels)
        ]
        train_tokenized_texts = [token_label_pair[0] for token_label_pair in train_tokenized_texts_and_labels]
        train_labels = [token_label_pair[1] for token_label_pair in train_tokenized_texts_and_labels]


        train_input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in train_tokenized_texts],
                          maxlen=max_length, dtype="long", value=0.0,
                          truncating="post", padding="post")


        # CoNLL has 9 labels (0-8) we add -1 as the PAD.
        train_tags = pad_sequences([[l for l in lab] for lab in train_labels], maxlen=max_length, value=-1, padding="post",
                     dtype="long", truncating="post")

        train_attention_masks = [[float(i != 0.0) for i in ii] for ii in train_input_ids]
        train_token_type_ids = [[0 for _ in range(max_length)] for _ in train_attention_masks]

        X_train = {
                'input_ids' : np.array(train_input_ids, dtype=np.int32)[:int((1-valid_fraction)*max_records_to_load)],
                'attention_mask' : np.array(train_attention_masks, dtype=np.int32)[:int((1-valid_fraction)*max_records_to_load)],
                'token_type_ids' : np.array(train_token_type_ids, dtype=np.int32)[:int((1-valid_fraction)*max_records_to_load)]
        }
        y_train = {'label' : np.array(train_tags, dtype=np.int32)[:int((1-valid_fraction)*max_records_to_load)]}
        
        X_valid = {
                'input_ids' : np.array(train_input_ids, dtype=np.int32)[int((1-valid_fraction)*max_records_to_load):max_records_to_load],
                'attention_mask' : np.array(train_attention_masks, dtype=np.int32)[int((1-valid_fraction)*max_records_to_load):max_records_to_load],
                'token_type_ids' : np.array(train_token_type_ids, dtype=np.int32)[int((1-valid_fraction)*max_records_to_load):max_records_to_load]
        }
        y_valid = {'label' : np.array(train_tags, dtype=np.int32)[int((1-valid_fraction)*max_records_to_load):max_records_to_load]}

        DATASETS['conll2003'] = {
            'X_train' : X_train,
            'y_train' : y_train,
            'X_valid' : X_valid,
            'y_valid' : y_valid
        }
    else:    
        X_train = DATASETS['conll2003']['X_train']
        y_train = DATASETS['conll2003']['y_train']
        X_valid = DATASETS['conll2003']['X_valid']
        y_valid = DATASETS['conll2003']['y_valid']


    X_train_batch, y_train_batch = sample_data('conll2003', X_train, y_train, int((1-valid_fraction) * batch_size), sampling_stratergy, model_path, custom_objects)

    validation_size = int(valid_fraction * batch_size)
    X_valid_batch = {k: X_valid[k][:validation_size] for k in X_valid}
    y_valid_batch = {k: y_valid[k][:validation_size] for k in y_valid}

    DATASETS['conll2003']['X_valid'] = {k: X_valid[k][validation_size:] for k in X_valid}
    DATASETS['conll2003']['y_valid'] = {k: y_valid[k][validation_size:] for k in y_valid}

    return  X_train_batch, y_train_batch, X_valid_batch, y_valid_batch
