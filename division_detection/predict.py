""" This module contains prediction related methods
"""

import os

from multiprocessing import Queue, Process
from warnings import warn
from time import time

import h5py

import numpy as np
import tensorflow as tf

from pathos.multiprocessing import ProcessPool as Pool

from division_detection.model import fetch_model
from division_detection.preprocessing import preprocess_vol
from division_detection.vol_preprocessing import in_mem_chunker, regular_chunker, fetch_vol_shape
from division_detection.utils import setup_logging

CHUNK_SIZE = (515, 500, 500)
STOP = 'STOP'

def pipeline_predict(model_name, chunk_size=(200, 150, 150), make_projs=False, plus_minus=2, run_all=False):
    """ Make predictions on all test, train and validation timepoints

    """
    if run_all:
        from division_detection.constants import NUM_TIMEPOINTS
        pred_tps = np.arange(3, NUM_TIMEPOINTS - 4)
    else:
        from division_detection.vol_preprocessing import fetch_annotations
        train_an = fetch_annotations('train')
        test_an = fetch_annotations('test')
        val_an = fetch_annotations('validation')
        used_tps = np.unique(np.concatenate([train_an, test_an, val_an])[:, 0]).astype(int)
        # make predictions for +- timestep as well
        pm_tps = [used_tps + pm for pm in range(-plus_minus, plus_minus + 1)]
        pred_tps = np.unique(np.concatenate(pm_tps)).astype(int)


    existing_tps = _existing_tps(model_name)
    pred_tps = [tp for tp in pred_tps if not tp in existing_tps]

    print("Found existing predictions for tps: {}".format(existing_tps))

    print("Submitting jobs for tps: {}".format(pred_tps))
    make_predictions_by_t_slurm_map(model_name, chunk_size, timepoints=pred_tps)

    print("Predictions complete, sparsifying")
    sparsify_predictions(model_name, timepoints=pred_tps)

    if make_projs:
        print("Running projections")
        from division_detection.analysis import pipeline_visualize
        pipeline_visualize(model_name)


def predict_from_inbox(model_name, in_dir, chunk_size=(200, 150, 150), allowed_gpus=[0]):
    """ Run predictions on all the files in in_dir

    Predictions will be saved to a directory in prediction_outbox with the same name as in_dir

    Args:
      model_name: name of model to use
      in_dir: dir containing the files to predict on
         must contain a dir named klb that contains the files themselves
      chunk_size: size of chunks to proces volume in
      allowed_gpus: CUDA ids of GPUs to use
         job will be parallelized across GPUs


    Results are saved to ~/results/model_name/dataset_name/dense
       where dataset_name is just the name of the folder containing the data

    """
    from division_detection.vol_preprocessing import write_klb_as_h5, save_bboxes_general

    # expected downstream
    if in_dir.endswith('/'):
        in_dir = in_dir[:-1]

    klb_dir = '{}/klb'.format(in_dir)
    h5_dir = '{}/h5'.format(in_dir)
    bbox_dir = '{}/bboxes'.format(in_dir)

    assert os.path.exists(klb_dir)
    os.mkdir(h5_dir)
    os.mkdir(bbox_dir)

    fnames = os.listdir(klb_dir)
    print("Found {} files in {}".format(len(fnames), klb_dir))

    print("Converting to h5...")
    write_klb_as_h5(klb_dir, h5_dir)

    print("Computing bounding boxes...")
    save_bboxes_general(h5_dir, bbox_dir)

    # now we can predi
    # extract the timepoints so we can figure out what we have to predict on

    timepoints = [int(fname.split('_')[1][:2]) for fname in fnames]
    tp_set = set(timepoints)
    valid_timepoints = []
    for t_idx in timepoints:
        # check that +- 3 exist
        t_rec_field = set([l_idx for l_idx in range(t_idx - 3, t_idx + 4)])
        # <= means containment here
        if t_rec_field <= tp_set:
            valid_timepoints.append(t_idx)

    make_predictions_by_t_local_map_general(
        model_name,
        in_dir,
        valid_timepoints,
        allowed_gpus=allowed_gpus,
        chunk_size=chunk_size
    )

def validate_predictions(model_name):
    """ Runs through dense predictions ands cleans corrupt files
    """
    pred_dir = '/nrs/turaga/bergera/division_detection/prediction_outbox/{}'.format(model_name)
    dense_pred_dir = '{}/dense'.format(pred_dir)

    corrupt_fnames = []
    for fname in os.listdir(dense_pred_dir):
        print("Checking {}".format(fname))
        try:
            with h5py.File('{}/{}'.format(dense_pred_dir, fname), 'r') as pred_file:
                # load something to make sure we can
                slice = pred_file['predictions'][10]
        except OSError as os_err:
            print("Caught os err on {}: {}".format(fname, os_err))
            corrupt_fnames.append(fname)

    print("Corrupt files: {}".format(corrupt_fnames))
    for fname in corrupt_fnames:
        os.remove('{}/{}'.format(dense_pred_dir, fname))

def validate_sparse_predictions(model_name):
    """ Runs through dense predictions ands cleans corrupt files
    """
    from division_detection.sparse_utils import load_coo_as_dense
    pred_dir = '/nrs/turaga/bergera/division_detection/prediction_outbox/{}'.format(model_name)
    sparse_pred_dir = '{}/sparse'.format(pred_dir)

    corrupt_fnames = []
    for fname in os.listdir(sparse_pred_dir):
        print("Checking {}".format(fname))
        try:
            pred = load_coo_as_dense('{}/{}'.format(sparse_pred_dir, fname))
        except OSError as os_err:
            print("Caught os err on {}: {}".format(fname, os_err))
            corrupt_fnames.append(fname)

    print("Corrupt files: {}".format(corrupt_fnames))
    for fname in corrupt_fnames:
        os.remove('{}/{}'.format(sparse_pred_dir, fname))




def make_predictions_at_t(model_name, t_predict, device='/gpu:0', in_mem=True, chunk_size=(200, 150, 150)):
    """ Helper functions, runs predictions for a single timepoint
    """
    with tf.device(device):
        print("Loading model")
        model, model_spec = fetch_model(model_name, device=device)


        print("beginning prediction")
        single_tp_nonblocking_predict(model, '{}.h5'.format(model_name), t_predict,
                                      device=device, in_mem=in_mem, chunk_size=chunk_size)

def make_predictions_by_t(model_name, device='/gpu:0', in_mem=True, chunk_size=(200, 150, 150)):
    """ Essentially calls make_predictions_at_t for every t, kinda
    """

    # fetch the number of timepoints
    from division_detection.vol_preprocessing import VOL_DIR_H5

    num_vols = len(os.listdir(VOL_DIR_H5))

    with tf.device(device):
        print("Loading model")
        model, model_spec = fetch_model(model_name, device=device)

        for t_predict in range(3, num_vols - 4):
            print("beginning prediction for {}".format(t_predict))
            single_tp_nonblocking_predict(model, '{}.h5'.format(model_name), t_predict,
                                          device=device, in_mem=in_mem, chunk_size=chunk_size)

def sparsify_predictions(model_name, timepoints=None):
    """ Computes and saves sparse representations for model predictions

    Args:
      model_name: name of model whose predictions you wish to sparsify
      timepoints: list of timepoints to sparsify - [int]
    """
    from ipp_tools.slurm import slurm_map
    from division_detection.constants import NUM_TIMEPOINTS

    pred_dir = '/nrs/turaga/bergera/division_detection/prediction_outbox/{}'.format(model_name)
    sparse_pred_dir = '{}/sparse'.format(pred_dir)

    if not os.path.exists(sparse_pred_dir):
        os.mkdir(sparse_pred_dir)

    existing_tps = set([int(fname[:-3]) for fname in os.listdir(sparse_pred_dir)])

    if timepoints is None:
        timepoints = np.arange(3, NUM_TIMEPOINTS - 4)

    timepoints = [t for t in timepoints if t not in existing_tps]


    def _sparsify_predictions_helper(t_idx):
        """ Helper function that sparsifies a single timepoint.
        """
        from division_detection.sparse_utils import save_dense_as_coo
        pred_dir = '/nrs/turaga/bergera/division_detection/prediction_outbox/{}'.format(model_name)
        dense_pred_dir = '{}/dense'.format(pred_dir)
        sparse_pred_dir = '{}/sparse'.format(pred_dir)

        if not os.path.exists('{}/{}.h5'.format(dense_pred_dir, t_idx)):
            warn('You asked me to sparsify predictions for {} but none exist'.format(t_idx))

        try:
            with h5py.File('{}/{}.h5'.format(dense_pred_dir, t_idx), 'r') as prediction_file:
                predictions = prediction_file['predictions']
                print("Loading ", t_idx)
                tp_preds = predictions[:]
                print("Saving ", t_idx)
                save_dense_as_coo(tp_preds, '{}/{}'.format(sparse_pred_dir, t_idx))
        except OSError as os_err:
            warn("Caught OS error while trying to read {}; continuing".format(t_idx))

    pool = Pool(20)
    pool.map(_sparsify_predictions_helper, timepoints)



def convert_to_klb(model_name, workers=10):
    from ipp_tools.slurm import slurm_map
    import pyklb
    pred_dir = '/nrs/turaga/bergera/division_detection/prediction_outbox/{}'.format(model_name)
    dense_pred_dir = '{}/dense'.format(pred_dir)
    klb_pred_dir = '{}/klb'.format(pred_dir)


    if not os.path.exists(klb_pred_dir):
        os.mkdir(klb_pred_dir)


    def _convert(fname):
        print("converting {}".format(fname))
        try:
            with h5py.File('{}/{}'.format(dense_pred_dir, fname), 'r') as pred_file:
                print("loading")
                predictions = pred_file['predictions'][:].squeeze()[np.newaxis, np.newaxis, ...]
                predictions = np.ascontiguousarray(predictions)
                print("writing")
                pyklb.writefull(predictions, '{}/{}.klb'.format(klb_pred_dir, fname[:-3]))
        except:
            print("Failed on {}".format(fname))

    pred_files = os.listdir(dense_pred_dir)
    resource_req = {
        'max_workers': workers,
        'worker_n_cpus': 10
    }

    slurm_map(_convert, pred_files, resource_req, env='tf_gpu2', job_name='klb_convert')



def make_predictions_by_t_local_map(model_name, chunk_size=(300, 150, 150), allowed_gpus=list(range(8)), timepoints=None):
    """ Make predictions for all timepoints, using all local gpus
    """
    from division_detection.vol_preprocessing import VOL_DIR_H5

    # fetch the number of timepoints
    num_vols = len(os.listdir(VOL_DIR_H5))
    if timepoints is None:
        timepoints = timepoints or np.arange(3,  num_vols - 4)
    n_gpus = len(allowed_gpus)
    split_timepoints = np.array_split(timepoints, n_gpus)
    devices = ['/gpu:{}'.format(idx) for idx in allowed_gpus]
    starred_args = list(zip(split_timepoints,
                            [model_name] * n_gpus,
                            [chunk_size] * n_gpus,
                            devices))

    def _star_helper(args):
        return _predict_local_helper(*args)

    print("Creating pool")
    pool = Pool(n_gpus)

    print("Dispatching jobs")
    pool.map(_star_helper, starred_args)

def make_predictions_by_t_slurm_map(model_name, chunk_size=(200, 150, 150),
                                    max_workers=16, min_workers=2,
                                    n_cpus=5, mem_mb=64000, timepoints=None):
    """ Make predictions for all timepoints, using all local gpus
    """
    from division_detection.vol_preprocessing import VOL_DIR_H5
    from ipp_tools.slurm import slurm_map

    # fetch the number of timepoints
    num_vols = len(os.listdir(VOL_DIR_H5))
    if timepoints is None:
        timepoints = timepoints or np.arange(3,  num_vols - 4)
    starred_args = list(zip(timepoints,
                            [model_name] * len(timepoints),
                            [chunk_size] * len(timepoints)))

    def _star_helper(args):
        return _slurm_predict_helper(*args)

    resource_req = {
        'max_workers': min(max_workers, len(timepoints)),
        'min_workers': min(min_workers, len(timepoints)),
        'worker_n_cpus': n_cpus,
        'worker_n_gpus': 1,
        'worker_mem_mb': mem_mb
    }

    slurm_map(_star_helper, starred_args, resource_req, env='tf_gpu2',
              job_name=model_name)


def make_predictions_by_t_slurm_map_general(model_name, in_dir, timepoints,
                                            chunk_size=(200, 150, 150),
                                            max_workers=64, min_workers=1,
                                            n_cpus=5, mem_mb=128000):
    """ Make predictions for all timepoints, using all local gpus
    """
    from division_detection.vol_preprocessing import VOL_DIR_H5
    from ipp_tools.slurm import slurm_map

    starred_args = list(zip(timepoints,
                            [in_dir] * len(timepoints),
                            [model_name] * len(timepoints),
                            [chunk_size] * len(timepoints)))

    def _star_helper(args):
        return _slurm_predict_helper_general(*args)

    resource_req = {
        'max_workers': min(max_workers, len(timepoints)),
        'min_workers': min(min_workers, len(timepoints)),
        'worker_n_cpus': n_cpus,
        'worker_n_gpus': 1,
        'worker_mem_mb': mem_mb
    }

    slurm_map(_star_helper, starred_args, resource_req, env='tf_gpu2',
              job_name=model_name, n_retries=100)


def make_predictions_by_t_local_map_general(model_name, in_dir, timepoints,
                                            allowed_gpus=[0],
                                            chunk_size=(200, 150, 150)):
    """ Make predictions for all timepoints, using all local gpus

    Args:
      model_name: name of model to use, will be looked up
      in_dir: absolute path to data dir
      timepoints: list of timepoints to process
      chunk_size: size of chunks to proces volume in
      allowed_gpus: CUDA ids of GPUs to use
         job will be parallelized across GPUs

    """
    n_gpus = len(allowed_gpus)
    split_timepoints = np.array_split(timepoints, n_gpus)
    devices = ['/gpu:{}'.format(idx) for idx in allowed_gpus]
    starred_args = list(zip(split_timepoints,
                            [in_dir] * n_gpus,
                            [model_name] * n_gpus,
                            [chunk_size] * n_gpus,
                            devices))


    def _star_helper(args):
        return _local_predict_helper_general(*args)


    print("Creating pool")
    pool = Pool(n_gpus)

    print("Dispatching jobs")
    pool.map(_star_helper, starred_args)


def make_predictions_by_t_ipp(model_name, ipp_profile='ssh_gpu_slowpoke4', chunk_size=(300, 150, 150)):
    """ Make predictions for all timepoint, distributing jobs using specified cluster which must be
    already started
    """
    from ipp_tools.mappers import gpu_job_runner
    from division_detection.vol_preprocessing import VOL_DIR_H5

    # fetch the number of timepoints
    num_vols = len(os.listdir(VOL_DIR_H5))
    timepoints = np.arange(3,  num_vols - 4)
    model_names = [model_name] * len(timepoints)
    chunk_sizes = [chunk_size] * len(timepoints)
    args_list = list(zip(model_names, timepoints, chunk_sizes))

    print("Submitting jobs, logging to ~/logs/division_detection/predict_{}".format(model_name))
    log_name = 'predict_{}'.format(model_name)
    log_dir = "~/logs/division_detection"

    gpu_job_runner(_predict_helper, args_list, ipp_profile=ipp_profile, log_name=log_name, log_dir=log_dir)


def single_tp_nonblocking_predict(model, predictions_name, t_predict,
                                  device='/gpu:0', in_mem=True, chunk_size=(200, 150, 150)):
    """ Runs predictions on a single timepoint.

    """
    pred_dir = '/nrs/turaga/bergera/division_detection/prediction_outbox/{}/dense'.format(predictions_name)

    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)

    log_path = os.path.expanduser('~/logs/division_detection')
    log_name = predictions_name + str(t_predict)
    logger = setup_logging(log_name, log_path)
    logger.info("Starting predictions at {}".format(t_predict))

    vol_shape = fetch_vol_shape()

    def predicter(predict_queue):
        if in_mem:
            # initialize chunker, on first call to next, vol stack will be loaded in mem
            chunk_gen = in_mem_chunker(t_predict, chunk_size=chunk_size)
        else:
            chunk_gen = regular_chunker(t_predict, chunk_size=chunk_size)
        # read from the queue as fast as possible and predict on it
        with tf.device(device):
            while True:
                print("fetching chunk")
                try:
                    chunk, coord = next(chunk_gen)
                    logger.info("predicting chunk at %s", coord)
                    prediction = model.predict(chunk[np.newaxis, ...])
                    predict_queue.put((prediction, coord))
                except StopIteration:
                    logger.info("Caught stop iteration. ")
                    predict_queue.put(STOP)
                    return


    def writer(predict_queue):
        try:
            # pull from the predict queue and write to disk
            with h5py.File('{}/{}.h5'.format(pred_dir, t_predict)) as prediction_file:
                predictions =  prediction_file.create_dataset('predictions', shape=vol_shape, dtype='f')

                chunk_idx = 0
                while True:
                    next_tok = predict_queue.get()
                    if next_tok != STOP:
                        prediction, chunk_coord = next_tok
                        logger.info("Writing chunk at %s", chunk_coord)
                        # correct for padding
                        pred_coord  = [chunk_coord[0] + 4,
                                       chunk_coord[1] + 22,
                                       chunk_coord[2] + 22]
                        pred_size = prediction.shape[2:]
                        predictions[pred_coord[0]: pred_coord[0] + pred_size[0],
                                    pred_coord[1]: pred_coord[1] + pred_size[1],
                                    pred_coord[2]: pred_coord[2] + pred_size[2]] = prediction[0]
                        chunk_idx += 1
                        # flush with some frequency for nilpotency reasons
                        if chunk_idx % 10 == 0:
                            prediction_file.flush()
                    else:
                        predict_queue.close()
                        predict_queue.join_thread()

        except Exception as general_err:
            logger.critical("Caught exception in writer: %s", general_err, type(general_err), t_predict)



    out_queue = Queue()

    logger.info("starting writer process")
    writer_p = Process(target=writer, args=((out_queue,)))
    writer_p.start()

    predicter(out_queue)

    logger.info("stopping writer process")
    writer_p.join()

def single_tp_nonblocking_predict_general(model, predictions_name, in_dir, t_predict,
                                          device='/gpu:0', chunk_size=(200, 150, 150)):
    """ Runs predictions on a single timepoint in a more general way

    """
    from division_detection.vol_preprocessing import general_regular_chunker

    assert in_dir[-1] != '/'
    ds_name = in_dir.split('/')[-1]

    pred_dir = '/results/{}/{}/dense'.format(predictions_name, ds_name)

    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)

    log_path = '/var/log/division_detection'

    log_name = predictions_name + str(t_predict)
    logger = setup_logging(log_name, log_path)
    logger.info("Starting predictions at {}".format(t_predict))

    h5_dir = in_dir + '/h5'
    ex_vol_path = '{}/{}'.format(h5_dir, os.listdir(h5_dir)[0])
    with h5py.File(ex_vol_path, 'r') as vol_file:
        vol_shape = vol_file['vol'].shape


    def predicter(predict_queue):
        chunk_gen = general_regular_chunker(t_predict, in_dir, chunk_size=chunk_size)
        # read from the queue as fast as possible and predict on it
        with tf.device(device):
            while True:
                print("fetching chunk")
                try:
                    chunk, coord = next(chunk_gen)
                    logger.info("predicting chunk at %s", coord)
                    prediction = model.predict(chunk[np.newaxis, ...])
                    predict_queue.put((prediction, coord))
                except StopIteration:
                    logger.info("Caught stop iteration. ")
                    predict_queue.put(STOP)
                    return


    def writer(predict_queue):
        try:
            # pull from the predict queue and write to disk
            with h5py.File('{}/{}.h5'.format(pred_dir, t_predict)) as prediction_file:
                predictions =  prediction_file.create_dataset('predictions', shape=vol_shape, dtype='f')

                chunk_idx = 0
                while True:
                    next_tok = predict_queue.get()
                    if next_tok != STOP:
                        prediction, chunk_coord = next_tok
                        logger.info("Writing chunk at %s", chunk_coord)
                        # correct for padding
                        pred_coord  = [chunk_coord[0] + 4,
                                       chunk_coord[1] + 22,
                                       chunk_coord[2] + 22]
                        pred_size = prediction.shape[2:]
                        predictions[pred_coord[0]: pred_coord[0] + pred_size[0],
                                    pred_coord[1]: pred_coord[1] + pred_size[1],
                                    pred_coord[2]: pred_coord[2] + pred_size[2]] = prediction[0]
                        chunk_idx += 1
                        # flush with some frequency for nilpotency reasons
                        if chunk_idx % 10 == 0:
                            prediction_file.flush()
                    else:
                        predict_queue.close()
                        predict_queue.join_thread()

        except Exception as general_err:
            logger.critical("Caught exception in writer: %s %s %s", general_err, type(general_err), t_predict)



    out_queue = Queue()

    logger.info("starting writer process")
    writer_p = Process(target=writer, args=((out_queue,)))
    writer_p.start()

    predicter(out_queue)

    logger.info("stopping writer process")
    writer_p.join()


def extract_coords(vol_path, min_sigma=1, max_sigma=50,
                   chunk_size=(200, 200, 200), overlap=50):
    """ Extract the centroid coordinates of the volume
    Current implementation will almost certainly label the same chunk multiple times

    Args:
      vol_path: path to klb file - str
      min_sigma: size of smallest blob - float
      max_sigma: size of largest blob - float
      chunk_size: size of chunks to process, must be as large as biggest cells - [3] - (z_dim, y_dim, x_dim)

    Returns:
      blob_info: array with info on the N detected blobs - [N, 5] - (x, y, z, sigma, confidence)
    """
    import pyklb
    from skimage.feature import blob_log

    print("Loading vol")
    vol = pyklb.readfull(vol_path)
    print("Loading volume with shape: {}".format(vol.shape))


    def run_vol(vol):
        start_time = time()
        print("Running blob detection")
        # NOTE: as of the writing, skimage still does not support Nd blob detection
        # use my fork instead: https://github.com/rueberger/scikit-image/tree/blob_nd
        # be sure to use the blob_nd branch
        detected_blobs = blob_log(vol, min_sigma=min_sigma, max_sigma=max_sigma)
        print("Blob detection completed in {} seconds".format(time() - start_time))
        return detected_blobs

    chunks, chunk_coords = zip(*list(chunk_generator(vol, overlap=overlap, chunk_size=chunk_size)))

    print("Mapping blob detection over {} chunks".format(len(chunks)))
    # by default spawns n_cpus workers
    pool = Pool()
    chunk_blobs = pool.map(run_vol, chunks)

    detected_blobs = []
    for chunk_blob, chunk_coords in zip(chunk_blobs, chunk_coords):
        if len(chunk_blob) == 0:
            continue

        # do not offset the size of the blob
        chunk_offset = np.concatenate([chunk_coords, [0]])
        detected_blobs.append(chunk_blob + chunk_offset)

    # [n_blobs, 4]
    detected_blobs = np.concatenate(detected_blobs, axis=0)

    if len(detected_blobs) == 0:
        print("No blobs detected, aborting")
        return None

    print("Extracting ")
    confidences = vol[detected_blobs[:, 0].astype(int),
                      detected_blobs[:, 1].astype(int),
                      detected_blobs[:, 2].astype(int)]

    # [n_blobs, 5]
    blob_data = np.concatenate([detected_blobs, confidences[:, np.newaxis]], axis=1)
    return blob_data


def run_coord_extract():
    """ Extract and save the coordinates of all files in the prediction_outbox
    """
    logger_name = '{}.coord_extract'.format(__name__)
    logger_path = os.path.expanduser('~/logs/division_detection/coord_extract.log')
    logger = setup_logging(logger_name, logger_path)
    prediction_path = '/nrs/turaga/bergera/division_detection/prediction_outbox'
    prediction_names = os.listdir(prediction_path)
    for prediction_name in prediction_names:
        if prediction_name.endswith('.klb'):
            # if the coordinates do not already exist
            if not '{}.csv'.format(prediction_name[:-4]) in prediction_names:
                logger.info("Starting coord extraction for %s", prediction_name)
                blob_data = extract_coords('{}/{}'.format(prediction_path, prediction_name))
                logger.info('Coord extraction complete')
                if blob_data is not None:
                    save_path = '{}/{}.csv'.format(prediction_path, prediction_name[:-4])
                    np.savetxt(save_path, blob_data, delimiter=',')
                    logger.info('Saved coords')
                else:
                    logger.warn('No blobs detected')


def fetch_chunks(vol_paths, chunk_size):
    """ Grabs a chunk from the first octant

    Args:
      vol_paths: list of paths to klb files ordered by t idx - list(str)
        must be of length 3
      chunk_size: size of chunk to take - [z_dim, y_dim, x_dim]

    Returns:
      chunked_vol: array - [z_dim', y_dim', x_dim', t_dim]

    """
    import pyklb
    assert len(vol_paths) == 3
    assert len(chunk_size) == 3

    vols = [pyklb.readfull(vol_path)[:chunk_size[0], :chunk_size[1], :chunk_size[2]] for vol_path in vol_paths]
    vol_stack = np.stack(vols, axis=-1)
    ds_vols = preprocess_vol(vol_stack)
    return ds_vols

def chunk_generator(vols, overlap=10, chunk_size=None):
    """ Iterates over vols using chunks of CHUNK_SIZE
    overlapping by overlap in all dimensions, except for possibly
    the last chunk in each dimension

    Args:
      vols: vols to chunk - [z_dim, y_dim, x_dim, 3] or  [z_dim, y_dim, x_dim]
      overlap:  amount that chunks should overlap
      chunk_size: (optional) chunk size - [3] - [z_dim, y_dim, x_dim]

    Returns
      chunk_gen: generator over chunks

      next returns:
        chunk: chunk of vol - CHUNK_SIZE + [3] or CHUNK_SIZE if vols.ndim == 3
        corner_coords: coordinate of the chunk corner with the closest to the origin - [3]: [z_coord, y_coord, x_coord]
    """
    from math import ceil

    chunk_size = chunk_size or CHUNK_SIZE

    n_chunks_by_dim = [ceil(vol_ax_len / float(chunk_ax_len - overlap)) for vol_ax_len, chunk_ax_len in zip(vols.shape[:3], chunk_size) ]

    for z_idx in range(n_chunks_by_dim[0]):
        z_coord = z_idx * (chunk_size[0] - overlap)
        if z_coord + chunk_size[0] >= vols.shape[0] and n_chunks_by_dim[0] > 1:
            z_coord = vols.shape[0] - chunk_size[0]
        for y_idx in range(n_chunks_by_dim[1]):
            y_coord = y_idx * (chunk_size[1] - overlap)
            if y_coord + chunk_size[1] >= vols.shape[1] and n_chunks_by_dim[1] > 1:
                y_coord = vols.shape[1] - chunk_size[1]
            for x_idx in range(n_chunks_by_dim[2]):
                x_coord = x_idx * (chunk_size[2] - overlap)
                if x_coord + chunk_size[2] >= vols.shape[2] and n_chunks_by_dim[2] > 1:
                    x_coord = vols.shape[2] - chunk_size[2]
                chunk = vols[z_coord: z_coord + chunk_size[0], y_coord: y_coord + chunk_size[1],  x_coord: x_coord + chunk_size[2]]
                chunk_coords = (z_coord, y_coord, x_coord)

                yield chunk, chunk_coords

# ==========================
#      PRIVATE METHODS
# ==========================


def _existing_tps(model_name):
    pred_dir = '/nrs/turaga/bergera/division_detection/prediction_outbox/{}/dense'.format(model_name)
    if not os.path.exists(pred_dir):
        return set([])
    existing_tp_fnames = os.listdir(pred_dir)
    return set([int(tp_fname[:-3]) for tp_fname in existing_tp_fnames])

def _predict_helper(main_args, device):
    from division_detection.predict import single_tp_nonblocking_predict
    from division_detection.model import fetch_model
    helper_model_name, helper_t_predict, helper_chunk_size = main_args
    model, model_spec = fetch_model(helper_model_name, device=device)
    single_tp_nonblocking_predict(model, "{}.h5".format(helper_model_name),
                                  helper_t_predict, device, in_mem=False,
                                  chunk_size=helper_chunk_size)

def _predict_local_helper(timepoints, model_name, chunk_size, device):
    model, model_spec = fetch_model(model_name, device=device)
    for t_predict in timepoints:
        single_tp_nonblocking_predict(model, "{}.h5".format(model_name),
                                  t_predict, device, in_mem=False,
                                  chunk_size=chunk_size)

def _local_predict_helper_general(timepoints, in_dir, model_name, chunk_size, device):
    model, model_spec = fetch_model(model_name, device=device)
    for t_predict in timepoints:
        single_tp_nonblocking_predict_general(model, model_name, in_dir,
                                              t_predict, device,
                                              chunk_size=chunk_size)


def _slurm_predict_helper(timepoint, model_name, chunk_size):
    model, model_spec = fetch_model(model_name, device='/gpu:0')
    single_tp_nonblocking_predict(model, model_name,
                                  timepoint, '/gpu:0', in_mem=False,
                                  chunk_size=chunk_size)

def _slurm_predict_helper_general(timepoint, in_dir, model_name, chunk_size):
    model, model_spec = fetch_model(model_name, device='/gpu:0')
    single_tp_nonblocking_predict_general(model, model_name, in_dir,
                                  timepoint, '/gpu:0',
                                  chunk_size=chunk_size)
