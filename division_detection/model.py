""" This module contains methods for constructing the model
"""


import tensorflow as tf

from keras.models import Sequential, Model
from keras.layers import Input, add, Activation
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution3D, ZeroPadding3D
from keras.layers.pooling import MaxPooling3D, AveragePooling3D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import concatenate
from keras.optimizers import get as get_optimizer
from keras.backend.tensorflow_backend import set_session
from keras import backend as K


import os
import json

import numpy as np

from division_detection.utils import package_path


def fetch_model(model_name, device='/gpu:0', compile_model=False):
    """ Fetches best performing model under that name

    Args:
      model_name: name model was saved under
      device: gpu to use - str
      compile_model: if True, compile loss and optimizer

    Returns:
       model: keras model with loaded weights
       model_spec:  corresponding model config dict
    """

    model_dir = '{}/model_zoo/{}'.format(package_path(), model_name)
    with open('{}/spec.json'.format(model_dir), 'r') as spec_file:
        model_spec = json.load(spec_file)

    if 'input_norm' not in model_spec['train_spec']:
        model_spec['train_spec']['input_norm'] = False

    with tf.device(device):
        gpu_options = tf.GPUOptions(allow_growth=True)
        config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
        sess = tf.Session(config=config)
        set_session(sess)

        model = build_model_mk2_full_res(model_spec, compile_model=compile_model, implicit_input_shape=False)

        weight_names = [fname for fname in os.listdir(model_dir) if fname.startswith('best_weights')]
        best_weight_name, loss = _fetch_best_weights(weight_names)

        print("Using weights with train loss: {}".format(loss))
        model.load_weights('{}/{}'.format(model_dir, best_weight_name))
        return model, model_spec




def generate_model_spec(model_name,
                        n_conv_filters,
                        n_output_hus=[64, 64],
                        residual=False,
                        clip_grads=False,
                        optimizer='adam',
                        activation='relu',
                        batch_norm=True,
                        batch_size=32,
                        early_stopping_patience=15,
                        lr_plateau_patience=5,
                        init_lr=0.001,
                        bulk_chunk_size=None,
                        max_bulk_chunk_size=None,
                        partial_weight=1e-4,
                        uniform_frac=1,
                        generation=7,
                        output_bn=True,
                        loss_weight='prior'):
    """ Generates a json specifying the model architecture and training hyperparams

    Args:
      model_name: used to save, should be unique - str
      n_conv_filters: number of filters to use in CNN layers - int or list of len 8
      n_output_hus: number of hus in the output layer - list
      residual: (optional) add residual connections
      receptive_field_size: implicit size of model receptive field - [t, x, y, z]
        all dimensions must be odd for centering purposes
        x,y dimensions must be 5x z dimension
      clip_grads: (optional) clip gradients before applying update
      optimizer: name of optimizer to use - str
      activation: activation to be used for all hidden layers - keras.layers.Activation or str
      batch_norm: whether to use batch normalization - bool
      uniform_frac: fraction of bulk samples to be drawn uniformly spatially

    Returns:
      model_spec: dict specifying model
    """
    assert isinstance(n_output_hus, list)

    if isinstance(n_conv_filters, int):
        n_conv_filters = [n_conv_filters] * 8
    else:
        assert isinstance(n_conv_filters, list)
        assert len(n_conv_filters) == 8

    assert bulk_chunk_size is not None
    if max_bulk_chunk_size is None:
        print("No max_bulk_chunk_size specified, using {}".format(bulk_chunk_size))
        max_bulk_chunk_size = bulk_chunk_size

    # NOTE: for now, clip_grad, batch_norm are double listed
    # eventually it will only exist in train_spec
    model_config = {
        'optimizer': optimizer,
        'clip_grad': clip_grads,
        'name': model_name,
        'residual': residual,
        'n_conv_filters': n_conv_filters,
        'n_output_hus': n_output_hus,
        'activation': activation,
        'batch_norm': batch_norm,
        'output_bn': output_bn,
        'generation': generation,
        'data_spec': {
            'uniform_frac': uniform_frac,
            'include_augment': True,
            'batch_size': batch_size,
            'bulk_chunk_size' : bulk_chunk_size,
            'max_bulk_chunk_size': max_bulk_chunk_size,
            'loss_weighter': loss_weight
        },
        'train_spec': {
            'early_stopping_patience': early_stopping_patience,
            'lr_plateau_patience': lr_plateau_patience,
            'learning_rate': init_lr,
            'clip_grads':  clip_grads,
            'partial_weight': partial_weight
        }
    }
    return model_config


def build_model_mk2_full_res(model_spec, n_replicas=1, compile_model=True, implicit_input_shape=False):
    """ Build the rull res version of the model, which has
    a receptive field of shape [7, 9, 45, 45]

    Args:
     model_spec: model spec as defined in generate_model_spec
     n_replicas: number of model replicas to build - int
       multiple replicas are used for combining gradients for batches with different shapes
       ie, instead of alternating batches, add a replica with the needed shape
     compile_model: if True compiles model before returning
     implicit_input_shape: if True, uses implicit input shape
       only for getting layer shapes, useless in practice


    """
    # model constants
    implicit_shape = [7, 9, 45, 45]
    xy_fws = [13, 11, 7, 9, 3, 3, 3, 3]
    z_fws = [1, 1, 1, 1, 3, 3, 3, 3]

    train_spec = model_spec['train_spec']

    if implicit_input_shape:
        inputs = [Input(implicit_shape) for _ in range(n_replicas)]
    else:
        inputs = [Input([7, None, None, None], name='output_{}'.format(idx)) for idx in range(n_replicas)]
    # list of lists, one for each replica
    layer_outs = [[inp] for inp in inputs]

    if 'output_bn' in model_spec:
        output_bn = model_spec['output_bn']
    else:
        if 'generation' in model_spec and model_spec['generation'] >= 6:
            output_bn = True
        else:
            output_bn = False

    # all the other activations can be looked up with Activation
    activation = model_spec['activation']
    if activation == 'LeakyReLU':
        activation = LeakyReLU()
    else:
        activation = Activation(activation)


    for z_fw, xy_fw, n_filter in zip(z_fws,
                                     xy_fws,
                                     model_spec['n_conv_filters']):
        if model_spec['batch_norm']:
            cnn_layer = Convolution3D(n_filter, [z_fw, xy_fw, xy_fw],
                                    data_format='channels_first', use_bias=False)
            bn_layer = BatchNormalization(axis=1)
            for replica_idx in range(n_replicas):
                replica_curr_inp = layer_outs[replica_idx][-1]
                cnn_out = cnn_layer(replica_curr_inp)
                bn_out = bn_layer(cnn_out)
                layer_outs[replica_idx].append(activation(bn_out))
        else:
            cnn_layer = Convolution3D(n_filter, [z_fw, xy_fw, xy_fw],
                                      data_format='channels_first', activation=activation)
            for replica_idx in range(n_replicas):
                replica_curr_inp = layer_outs[replica_idx][-1]
                cnn_out = cnn_layer(replica_curr_inp)
                layer_outs[replica_idx].append(cnn_out)

    # output shape is  now [1, 1, 1, n_conv_filters] for input shape equal to receptive field size
    # The output layer, collapses filters into a binary prediction
    output_acts = [activation] * len(model_spec['n_output_hus']) + [Activation('sigmoid')]
    for n_hu, act in zip(model_spec['n_output_hus'] + [1], output_acts):
        if output_bn:
            cnn_layer = Convolution3D(n_hu, [1, 1, 1], data_format='channels_first', use_bias=False)
            bn_layer = BatchNormalization(axis=1)
        else:
            cnn_layer = Convolution3D(n_hu, [1, 1, 1], data_format='channels_first', activation=act)
        for replica_idx in range(n_replicas):
            if model_spec['residual'] and layer_outs[replica_idx][-1].shape == layer_outs[replica_idx][-2].shape:
                curr_input = add([layer_outs[replica_idx][-1], layer_outs[replica_idx][-2]])
            else:
                curr_input = layer_outs[replica_idx][-1]

            if output_bn:
                cnn_out = cnn_layer(curr_input)
                bn_out = bn_layer(cnn_out)
                layer_outs[replica_idx].append(act(bn_out))
            else:
                layer_outs[replica_idx].append(cnn_layer(curr_input))

    model = Model(inputs=inputs, outputs=[replica_outs[-1] for replica_outs in layer_outs])

    if compile_model:
        opt_kwargs = {'lr': train_spec['learning_rate']}

        if train_spec['clip_grads']:
            opt_kwargs.update({'clipnorm': 1., 'clipvalue': 0.5})

        # get_optimizer returns an instance, type gets the class
        optimizer = type(get_optimizer(model_spec['optimizer']))(**opt_kwargs)

        loss = loss_factory(model_spec)

        if n_replicas > 1:
            # second replica is always used for partials
            model.compile(loss=loss, optimizer=optimizer, loss_weights=[1, train_spec['partial_weight']])
        else:
            model.compile(loss=loss, optimizer=optimizer)
    return model

def _fetch_best_weights(weight_names):
    """ Parse the weight names and pick the one with the smallest validation loss
    """
    losses = []
    for name in weight_names:
        pre, post = name.split('.')[1:3]
        loss_str = pre.split('-')[-1] + '.' + post
        losses.append(float(loss_str))
    if len(losses) == 0:
        return None, 1000
    min_idx = np.argmin(losses)
    min_val = losses[min_idx]
    losses = np.asarray(losses)
    # handle multiplicity by preferring later weights
    if len(losses[losses == min_val]) > 1:

        return sorted([wname for wname, loss in zip(weight_names, losses) if loss == min_val])[-1], min_val
    else:
        return weight_names[min_idx], losses[min_idx]

def loss_factory(model_spec):
    """ Construct and return a loss according to the model spec
    """
    data_spec = model_spec['data_spec']
    reweighter_name = data_spec['loss_weighter']

    if reweighter_name == 'prior':
        reweighter = prior_reweight
    elif reweighter_name == 'batch':
        reweighter = batch_reweight
    elif reweighter_name == 'ones':
        reweighter = ones_reweight
    elif reweighter_name == 'batch_unbounded':
        reweighter = unbounded_batch_reweight
    else:
        raise NotImplementedError("Unrecognized option for reweighter: {}".format(reweighter_name))


    def balanced_binary_crossentropy(y_true, y_pred):
        raw_xh = K.binary_crossentropy(y_pred, y_true)
        reweights = reweighter(y_true)
        weighted_xh = raw_xh * reweights
        return K.mean(weighted_xh, axis=-1)

    return balanced_binary_crossentropy

def ones_reweight(y_true):
    """ Null reweighter. Just returns vector of ones
    """
    return tf.ones_like(y_true)


def prior_reweight(y_true):
    """ Constant reweight using the class prior
    """
    from division_detection.constants import PRIOR_P_DIV
    pos_reweights  = tf.ones_like(y_true) * (1 / PRIOR_P_DIV)
    neg_reweights =  tf.ones_like(y_true) * (1 / (1 - PRIOR_P_DIV))
    reweights = tf.where(y_true == 1, x=pos_reweights, y=neg_reweights)
    return reweights

def batch_reweight(y_true, reweight_range=(1e-2, 1e2)):
    """ Reweight using batch statistics
    """
    pos_frac = tf.reduce_sum(y_true) / tf.reduce_prod(tf.to_float(tf.shape(y_true)))

    def has_division():
        pos_reweights = tf.ones_like(y_true) * (1 / pos_frac)
        neg_reweights = tf.ones_like(y_true) * (1 / (1 - pos_frac))
        reweights = tf.where(y_true == 1, x=pos_reweights, y=neg_reweights)
        return reweights

    def no_division():
        reweights = tf.ones_like(y_true)
        return reweights

    reweights = tf.cond(pos_frac > 0, has_division, no_division)
    reweights = tf.clip_by_value(reweights, reweight_range[0], reweight_range[1])
    return reweights

def unbounded_batch_reweight(y_true):
    """ Reweight using batch statistics, no clipping weights
    """
    pos_frac = tf.reduce_sum(y_true) / tf.reduce_prod(tf.to_float(tf.shape(y_true)))

    def has_division():
        pos_reweights = tf.ones_like(y_true) * (1 / pos_frac)
        neg_reweights = tf.ones_like(y_true) * (1 / (1 - pos_frac))
        reweights = tf.where(y_true == 1, x=pos_reweights, y=neg_reweights)
        return reweights

    def no_division():
        reweights = tf.ones_like(y_true)
        return reweights

    reweights = tf.cond(pos_frac > 0, has_division, no_division)
    return reweights
