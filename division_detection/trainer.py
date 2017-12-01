""" This module contains methods for training the model
"""

import tensorflow as tf
import os
import json

from warnings import warn

from division_detection.model import build_model_mk2_full_res
from division_detection.preprocessing import batch_generator
from division_detection.vol_preprocessing import fetch_callbacks, pipeline_batch_generator
from division_detection.utils import clean_dir
from division_detection.constants import VAL_TIMEPOINTS

from keras.backend.tensorflow_backend import set_session
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau
import h5py


def pipeline_train(model_spec, epoch_size=2048, n_epochs=250, n_workers=22, device='/gpu:0'):
    """ Pipelined train method
    Does everything

    Args:
      model: compiled keras model
      model_spec: model spec as returned by model.generate_model_spec
    """
    warn("Training not supported by this release... check back later...")
    save_dir = '/groups/turaga/home/bergera/results/div_detect/model/mk2/{}'.format(model_spec['name'])
    print("Save dir: {}".format(save_dir))

    print("Fetching callbacks")
    callbacks = fetch_callbacks(model_spec)

    # save model spec
    with open('{}/spec.json'.format(save_dir), 'w') as spec_file:
        json.dump(model_spec, spec_file)


    with tf.device(device):
        gpu_options = tf.GPUOptions(allow_growth=True)
        config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
        sess = tf.Session(config=config)
        set_session(sess)

        print("Building model")
        model = build_model_mk2_full_res(model_spec, n_replicas=2)

        batch_gen = pipeline_batch_generator(model_spec)

        # TODO: fix size of validation set
        valid_spec = model_spec.copy()
        valid_gen = pipeline_batch_generator(valid_spec, mode='validation')

        print("Here we go")
        model.fit_generator(batch_gen, epoch_size, epochs=n_epochs,
                            validation_data=valid_gen, validation_steps=512,
                            callbacks=callbacks, workers=n_workers, pickle_safe=True)





def train_model(vgg=False, batch_size=32, epoch_size=1024, n_epochs=250,
                loss='binary_crossentropy', save=True, **kwargs):
    """ Load, compile, and fit the model

    Args:
       vgg: (optional) if True, use vgg style model
       batch_size: (optional) number of samples per batch - int
       epoch_size: (optional) number of samples per epoch - int
       n_epochs: (optional) maximum number of epochs - int
       loss: (optional) loss to use - str
       save: (optional) if true, save model
       slideable: (optional) use slideable model

    additional kwargs are passed to bugild_model

    Returns:
       model: the trained model
    """
    results_dir = os.path.expanduser('~/results/div_detect')
    if save:
        if vgg:
            print("Training vgg model")
            log_dir = '{}/logs/vgg'.format(results_dir)
            model_dir = '{}/model/vgg'.format(results_dir)
        else:
            print("Training purely convolutional model")
            log_dir = '{}/logs/conv'.format(results_dir)
            model_dir = '{}/model/conv'.format(results_dir)

        clean_dir(log_dir)
        clean_dir(model_dir)

    print("Loading model")
    model = build_model(vgg=vgg, **kwargs)

    print("Compiling model")
    model.compile(optimizer='adam',
                  loss=loss,
                  metrics=['accuracy'])

    train_gen = batch_generator('train', batch_size=batch_size, slideable_shape=True)
    valid_gen = batch_generator('valid', batch_size=batch_size, slideable_shape=True)

    print("Fitting model")
    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_options)
    sess = tf.Session(config=config)
    set_session(sess)

    callbacks = [
        # stop training if val loss does not improve for 50 epochs
        EarlyStopping(patience=50,
                      verbose=1
        ),
        # cuts learning rate in half when val loss stagnates
        ReduceLROnPlateau(factor=0.5,
                          patience=10,
                          cooldown=25,
                          verbose=1
        )
    ]

    if save:
        callbacks.extend([
            ModelCheckpoint(model_dir + '/best_weights.{epoch:02d}-{val_loss:.2f}.h5',
                            verbose=1,
                            period=25,
                            save_weights_only=True,
                            save_best_only=True
            ),
            # computes histograms of layer activations every 10 epochs
            TensorBoard(log_dir=log_dir,
                        write_graph=False,
                        histogram_freq=25
            )
        ])
    # unclear if increasing the number of workers helps at all
    history = model.fit_generator(train_gen, epoch_size, n_epochs,
                                  callbacks=callbacks,
                                  validation_data=valid_gen,
                                  nb_val_samples=512,
                                  max_q_size=6400
    )
    return model


def set_up():
    """ This only needs to be run once
    """
    from division_detection.preprocessing import save_holdout_split, process_and_save
    process_and_save()
    save_holdout_split()


def spearmint_objective(model_name, job_id, params):
    """ Train up the model and return the validation loss
    """
    vgg = model_name == 'vgg'
    n_conv_hu = params['n_conv_hu'][0]

    search_dir = os.path.expanduser('~/results/div_detect/search/{}'.format(model_name))
    validation_data_path = os.path.expanduser('~/data/div_detect/valid_recs.h5')

    model = train_model(vgg=vgg, n_conv_hidden_units=n_conv_hu, save=False)

    with h5py.File(validation_data_path, 'r') as valid_file:
        valid_data = valid_file['records'][:]
        valid_labels = valid_file['labels'][:]

    validation_loss = model.evaluate(valid_data, valid_labels)[0]

    with open("{}/loss/job_{}.json".format(search_dir, job_id), 'w') as hist_file:
        job_rec = {
            'loss': validation_loss,
            'n_conv_hu': n_conv_hu
        }
        json.dump(job_rec, hist_file)

    return validation_loss
