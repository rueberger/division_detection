""" This module contains preprocessing scripts for extracting and augmenting data from fully marked up volumes
"""

import numpy as np
import os
import math
import logging

import h5py
import augment

from math import ceil, floor

import json

from scipy.io import loadmat
from sklearn.neighbors import NearestNeighbors

from warnings import warn

from division_detection.preprocessing import rot90

from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint, ProgbarLogger, ReduceLROnPlateau

POS_LABELS = {1, 2, 3, 4, 5, 103}
ANNOTATIONS_PATH = os.path.expanduser('~/data/div_detect/annotations')
FULL_ANNOTATIONS_PATH = '{}/complete/full_volume_div_annotations.csv'.format(ANNOTATIONS_PATH)
PARTIAL_ANNOTATIONS_PATH = '{}/partial/divisionAnnotations.mat'.format(ANNOTATIONS_PATH)
VALIDATION_ANNOTATIONS_PATH = '{}/validation/annotations.csv'.format(ANNOTATIONS_PATH)
CORRECTED_ANNOTATIONS_PATH = '{}/fp_corrections'.format(ANNOTATIONS_PATH)
VOL_DIR = '/nrs/turaga/bergera/division_detection/full_raw_timeseries'
VOL_DIR_H5 = '/nrs/turaga/bergera/division_detection/full_h5_timeseries'
STACK_PATH = os.path.expanduser('~/data/div_detect/vol_stacks/stacks.h5')
PARTIAL_CUTOUTS_PATH = os.path.expanduser('~/data/div_detect/partial_cutouts/all.h5')
SPLIT_PARTIALS_PATH_TEMPLATE = os.path.expanduser('~/data/div_detect/partial_cutouts/{}.h5')
GT_TARGET_PATH = os.path.expanduser('~/data/div_detect/gt_vols/train_target.h5')
FULL_RES_GT_TARGET_PATH = os.path.expanduser('~/data/div_detect/full_res_gt_vols/train_target.h5')
BBOXES_PATH = os.path.expanduser('~/data/div_detect/bboxes/bboxes.json')
REC_FIELD_SHAPE = [7, 45, 45, 9]

# the timepoints in the 'validation' file to be reserved as validation data
VAL_TIMEPOINTS = [250]
TEST_TIMEPOINTS = [120, 240, 360]

def pipeline_batch_generator(model_spec, mode='train'):
    """ Thin wrapper around batch_generator
    Parses arguments from model_spec for integration into the pipeline

    Args:
      model_spec: dict as returned by model.generate_model_spec
      mode: one of ['train', 'test', 'validation']


    Generator returns:
      batch_cutouts: batch of cutouts - [batch_size, t, x, y, z]
      batch_labels: batch of labels - [batch_size]

    """
    if mode not in ['train', 'test', 'validation']:
        raise NotImplementedError("Unrecognized mode: {}".format(mode))

    data_spec = model_spec['data_spec']

    # TODO: option to not run in replica mode

    if mode == 'train':
        bulk_batch_gen = bulk_batch_generator(data_spec['bulk_chunk_size'],
                                              data_spec['batch_size'],
                                              p_uniform=data_spec['uniform_frac'],
                                              mode=mode)


        rec_field_shape = [7, 45, 45, 9]
        partial_gen = partial_batch_generator(rec_field_shape,
                                              data_spec['batch_size'],
                                              data_spec['include_augment'])

        while True:
            bulk_cutouts, bulk_targets = next(bulk_batch_gen)
            partial_cutouts, partial_targets = next(partial_gen)
            yield [bulk_cutouts, partial_cutouts], [bulk_targets, partial_targets]

    elif mode == 'validation':
        bulk_batch_gen = bulk_batch_generator(data_spec['bulk_chunk_size'],
                                              data_spec['batch_size'],
                                              p_uniform=data_spec['uniform_frac'],
                                              mode=mode)
        # draw partials randomly from the bulk batch
        while True:
            bulk_cutouts, bulk_targets = next(bulk_batch_gen)
            partial_cutouts = bulk_cutouts[:, :, :9, :45, :45]
            partial_targets = bulk_targets[:, :,  0, 0, 0][..., np.newaxis, np.newaxis, np.newaxis]
            yield [bulk_cutouts, partial_cutouts], [bulk_targets, partial_targets]

    else:
        bulk_batch_gen = bulk_batch_generator(data_spec['bulk_chunk_size'],
                                              data_spec['batch_size'],
                                              p_uniform=data_spec['uniform_frac'],
                                              mode=mode)
        while True:
            yield next(bulk_batch_gen)


def bulk_batch_generator(cutout_shape, batch_size, p_uniform=1, mode='train'):
    """ Batcher for bulk generators

    Args:
      cutout_shape: size of cutout to make, dimensions must be odd so it can be centered - [3] - (z, y, x)
      batch_size: size of batches to yield - int
      include_augment: if True, include augmentations - bool
      p_uniform: probability sample is drawn uniformly spatially
       otherwise drawn near a division
      mode: one of ['train', 'test', 'validation']

    """
    bulk_gen = balanced_bulk_generator(cutout_shape, mode=mode, p_uniform=p_uniform)

    while True:
        batch_cutouts = []
        batch_targets = []
        for _ in range(batch_size):
            cutout, target = next(bulk_gen)
            batch_cutouts.append(cutout)
            batch_targets.append(target)
        batch_cutouts = np.stack(batch_cutouts)
        batch_targets = np.stack(batch_targets)[:, np.newaxis, ...]
        yield batch_cutouts, batch_targets

def partial_batch_generator(partial_cutout_shape, batch_size, include_augment):
    """ Yields batches from partial cutouts. Handles augmentation

    Args:
      partial_cutout_shape: shape of partial cutouts, aka model receptive field - [t, x, y, z]
      batch_size:size of batches to yield - int
      include_augment: if True, include augmentations - bool
    """
    # TODO: select larger cutout shape for augment sub select
    sample_gen = _partial_generator(partial_cutout_shape)
    while True:
        batch_cutouts = []
        batch_targets = []
        for _ in range(batch_size):
            target, cutout = next(sample_gen)
            if include_augment:
                batch_cutouts.append(augment_vol(cutout))
            else:
                batch_cutouts.append(cutout)

            batch_targets.append(target)
        batch_cutouts = np.stack(batch_cutouts)
        batch_targets = np.stack(batch_targets).astype(np.float64)
        yield batch_cutouts, batch_targets


def _partial_generator(partial_cutout_shape):
    with h5py.File(PARTIAL_CUTOUTS_PATH, 'r') as partial_file:
        cutout_str = str(tuple(partial_cutout_shape))
        assert cutout_str in partial_file

        cutouts = partial_file['{}/cutouts'.format(cutout_str)]
        labels = partial_file['{}/labels'.format(cutout_str)]

        while True:
            idx_roll = np.random.randint(cutouts.shape[0])
            cutout_roll = cutouts[idx_roll]
            is_division = labels[idx_roll].reshape(1, 1, 1, 1)
            yield is_division, cutout_roll


def batch_generator(cutout_shape, batch_size, pos_weight=1., **kwargs):
    """ Generator over batches of cutouts

    Args:
      cutout_shape: size of cutout to make, dimensions must be odd so it can be centered - [4] - (t, x, y, z)
      batch_size: size of batches to yield - int
      pos_weight: (optional) change to give positive samples more weight

    Generator returns:
      batch_cutouts: batch of cutouts - [batch_size, t, x, y, z]
      batch_labels: batch of labels - [batch_size]

    """
    cutout_gen = cutout_generator(cutout_shape, **kwargs)
    while True:
        batch_cutouts = []
        batch_labels = []
        sample_weights = []
        for _ in range(batch_size):
            is_division, cutout = next(cutout_gen)
            if is_division:
                batch_labels.append(1)
                sample_weights.append(pos_weight)
            else:
                batch_labels.append(0)
                sample_weights.append(1.)
            batch_cutouts.append(cutout)
        batch_cutouts = np.stack(batch_cutouts)
        batch_labels = np.stack(batch_labels).reshape(-1, 1, 1, 1, 1)
        sample_weights = np.stack(sample_weights)
        yield batch_cutouts, batch_labels, sample_weights



def large_cutout_generator(cutout_shape):
    """ Yield large cutouts

    Args:
      cutout_shape: size of cutout to make - [z, y, x]

    """
    # verify shape
    assert len(cutout_shape) == 3

    # x_len, y_len must be divisible by 5 for integral downsampling
    assert cutout_shape[1] % 5 == 0
    assert cutout_shape[2] % 5 == 0

    # [z, y, x]
    output_cutout_shape = [
        cutout_shape[0] - 16,
        int(cutout_shape[1] / 5) - 16,
        int(cutout_shape[2] / 5) - 16
    ]

    # [z, y, x]
    vol_shape = fetch_vol_shape()

    # load annotations
    full_annotations = fetch_full_annotations()
    annotated_tps = np.unique(full_annotations[:, 0]).astype(np.int32)

    # load bboxes
    bboxes = fetch_bboxes()

    with h5py.File(STACK_PATH, 'r') as stack_file, h5py.File(GT_TARGET_PATH, 'r') as gt_file:

        while True:
            #  draw a timepoint
            t_roll = np.random.choice(annotated_tps)
            bbox = bboxes[t_roll]

            # choose coordinate for chunk corner
            # y and x are chosen to be integer multiples of 5
            # [z, y, x]
            inp_space_chunk_coord = [
                np.random.randint(max(0, bbox[0][0]), min(vol_shape[0] - cutout_shape[0], bbox[0][1])),
                np.random.randint(max(0, bbox[1][0]), min(floor((vol_shape[1] - cutout_shape[1]) / 5), bbox[1][1])) * 5,
                np.random.randint(max(0, bbox[2][0]), min(floor((vol_shape[2] - cutout_shape[2]) / 5), bbox[2][1])) * 5
            ]

            out_space_chunk_coord =  [
                inp_space_chunk_coord[0] + 8,
                int(inp_space_chunk_coord[1] / 5) + 8,
                int(inp_space_chunk_coord[2] / 5) + 8
            ]

            inp_cutout = stack_file[str(t_roll)][
                :,
                inp_space_chunk_coord[0]: inp_space_chunk_coord[0] + cutout_shape[0],
                inp_space_chunk_coord[1]: inp_space_chunk_coord[1] + cutout_shape[1],
                inp_space_chunk_coord[2]: inp_space_chunk_coord[2] + cutout_shape[2]
            ]

            target_cutout = gt_file[str(t_roll)][
                out_space_chunk_coord[0]: out_space_chunk_coord[0] + output_cutout_shape[0],
                out_space_chunk_coord[1]: out_space_chunk_coord[1] + output_cutout_shape[1],
                out_space_chunk_coord[2]: out_space_chunk_coord[2] + output_cutout_shape[2]
            ]

            yield target_cutout, inp_cutout


def balanced_bulk_generator(cutout_shape, offsets=(4, 22, 22),
                            perturb_range=(-5, 5), mode='train', debug=False, p_uniform=1):
    """ Class balanced cutout generator

    If cutout shape is too large it's not really possible to do class balance

    Half of the samples are chosen from the neighborhood of a division, the other half
     are drawn uniformly at random


    Args:
      cutout_shape: [z, y, x] - all ax lens must be add
      offsets: [z, y, x]
      perturb_range: [min, max] - range of perturbations that will be applied to positive chunks
      mode: one of ['train', 'test', 'validation']
      debug: if True use debug mode
      p_uniform: probability sample is drawn uniformly spatially at random

    """
    # verify shape
    assert len(cutout_shape) == 3

    # [z, y, x]
    vol_shape = fetch_vol_shape()

    # load annotations
    annotations = fetch_annotations(mode)
    annotated_tps = np.unique(annotations[:, 0]).astype(np.int32)

    vol_fetcher = vol_fetcher_factory(cutout_shape, offsets=offsets, debug=debug)

    with h5py.File(STACK_PATH, 'r') as stack_file, h5py.File(FULL_RES_GT_TARGET_PATH, 'r') as gt_file:
        while True:
            roll = np.random.random()
            # draw a sample in the neighborhood of a division
            if roll < (1 - p_uniform):
                t_roll = np.random.choice(annotated_tps)
                # [t, z, y, x]
                coord_roll = _draw_coord(annotations, t_roll)
                # add perturbation
                coord_roll = [ax_roll + np.random.randint(perturb_range[0], perturb_range[1] + 1) for ax_roll in coord_roll[1:]]

                yield vol_fetcher(t_roll, coord_roll, stack_file, gt_file)
            # draw a fair uniform sample
            else:
                t_roll = np.random.choice(annotated_tps)
                bbox = fetch_bb_at_t(t_roll)

                coord_roll = [np.random.randint(max(0, bb[0]), min(ax_len - cutout_len, bb[1])) for ax_len, cutout_len, bb in zip(vol_shape, cutout_shape, bbox)]

                yield vol_fetcher(t_roll, coord_roll, stack_file, gt_file)


def large_full_res_cutout_generator(cutout_shape, offsets=(4, 22, 22), mode='train'):
    """ Yields large cutouts
    Returns target chunk at full res

    Args:
      cutout_shape: [z, y, x]
      offsets: [z, y, x]
      mode: one of ['train', 'test', 'validation']
    """
    # verify shape
    assert len(cutout_shape) == 3

    # [z, y, x]
    output_cutout_shape = [ax_len - 2 * ax_offset for ax_len, ax_offset in zip(cutout_shape, offsets)]

    # [z, y, x]
    vol_shape = fetch_vol_shape()

    # load annotations
    annotations = fetch_annotations(mode)
    annotated_tps = np.unique(annotations[:, 0]).astype(np.int32)

    with h5py.File(STACK_PATH, 'r') as stack_file, h5py.File(FULL_RES_GT_TARGET_PATH, 'r') as gt_file:
        while True:
            #  draw a timepoint
            t_roll = np.random.choice(annotated_tps)
            bbox = fetch_bb_at_t(t_roll)

            # choose coordinate for chunk corner
            # y and x are chosen to be integer multiples of 5
            # [z, y, x]
            inp_space_chunk_coord = [np.random.randint(max(0, bb[0]), min(ax_len - cutout_len, bb[1])) for ax_len, cutout_len, bb in zip(vol_shape, cutout_shape, bbox)]

            out_space_chunk_coord = [in_coord + offset for in_coord, offset in zip(inp_space_chunk_coord, offsets)]

            inp_cutout = stack_file[str(t_roll)][
                :,
                inp_space_chunk_coord[0]: inp_space_chunk_coord[0] + cutout_shape[0],
                inp_space_chunk_coord[1]: inp_space_chunk_coord[1] + cutout_shape[1],
                inp_space_chunk_coord[2]: inp_space_chunk_coord[2] + cutout_shape[2]
            ]

            target_cutout = gt_file[str(t_roll)][
                out_space_chunk_coord[0]: out_space_chunk_coord[0] + output_cutout_shape[0],
                out_space_chunk_coord[1]: out_space_chunk_coord[1] + output_cutout_shape[1],
                out_space_chunk_coord[2]: out_space_chunk_coord[2] + output_cutout_shape[2]
            ]

            yield target_cutout, inp_cutout

def vol_fetcher_factory(output_shape, offsets=(4, 22, 22), debug=False):
    """ Abstracts away from chunk fetching logic and handles continuous augmentation

    Args:
      output_shape: [z, y, x]

    Returns:
      vol_fetcher(centroid_coord) -> cutout
    """
    assert output_shape[1] == output_shape[2]

    # [z, y, x]
    supersel_shape = list(continuous_augment_shape_transform(output_shape))
    supersel_offset = continuous_augment_offset(output_shape[1])

    # [z, y, x]
    vol_shape = np.array(fetch_vol_shape())
    far_boundary = vol_shape

    def sample_fetcher(t_idx, centroid_coord, stack_file, gt_file):
        """ Cutout from vol at specified location, zero padding as necessary

        Args:
          t_idx: timepoint to select from - int
          centroid_coord: [z, y, x]
          stack_file: open h5 file with vol stacks
          gt_file: open h5 file with gt renders
        """
        # NOTE: i think that it's going to be easiest to handle both separately after all
        # otherwise risk having different corner offsets etc
        # define corner offset and other things using full volume shape and
        # just adjust ground truth by padding offsets
        assert len(centroid_coord) == 3


        # NOTE: in *output* frame!n
        corner_offset = [int(ax_len / 2) for ax_len in output_shape]
        corner_coord = np.asarray([coord - offset for coord, offset in zip(centroid_coord, corner_offset)])
        corner_coord = continuous_augment_coord_transform(corner_coord, output_shape[1])

        supersel = np.zeros([7] + supersel_shape)
        gt_supersel = np.zeros(supersel_shape)

        # TODO: handle out of bounds coordinates better
        # too much zero padding will causes some zeros to be rotated in
        # may just have to reject those coords

        # lower cutout boundary
        coord_lb = [max(0, coord) for coord in corner_coord]
        # upper cutout boundary
        coord_ub = [min(coord + ax_len, upper_boundary) for coord, ax_len, upper_boundary in zip(corner_coord, supersel_shape, far_boundary)]

        pad_lb = np.abs(coord_lb - corner_coord)
        pad_ub = np.abs(np.asarray(coord_ub) - corner_coord)

        supersel[:,
                 pad_lb[0]:pad_ub[0],
                 pad_lb[1]:pad_ub[1],
                 pad_lb[2]:pad_ub[2]] = stack_file[str(t_idx)][:,
                                                                coord_lb[0]: coord_ub[0],
                                                                coord_lb[1]: coord_ub[1],
                                                                coord_lb[2]: coord_ub[2]]

        gt_supersel[pad_lb[0]:pad_ub[0],
                    pad_lb[1]:pad_ub[1],
                    pad_lb[2]:pad_ub[2]] = gt_file[str(t_idx)][coord_lb[0]: coord_ub[0],
                                                                coord_lb[1]: coord_ub[1],
                                                                coord_lb[2]: coord_ub[2]]

        rotation = np.random.random() * 2 * math.pi
        flip_z = np.random.random() < 0.5
        # TODO: add debug option with no augment?
        aug_supersel = continuous_augment(supersel, rotation, flip_z)
        aug_gt_supersel = continuous_augment(gt_supersel, rotation, flip_z)

        aug_vol = aug_supersel[:,
                               :,
                               supersel_offset: supersel_offset + output_shape[1],
                               supersel_offset: supersel_offset + output_shape[2]]

        aug_gt = aug_gt_supersel[offsets[0]:-offsets[0],
                                 supersel_offset + offsets[1]: supersel_offset + output_shape[1] - offsets[1],
                                 supersel_offset + offsets[2]: supersel_offset + output_shape[2] - offsets[2]]
        if debug:
            return aug_vol, aug_gt, aug_supersel, aug_gt_supersel, supersel, gt_supersel
        else:
            return aug_vol, aug_gt

    return sample_fetcher

def cutout_generator(cutout_shape, pos_radius=5, rotations=False,
                     partial_frac=0.05, sharpen_frac=0.5,
                     verbose=False, include_augment=False,
                     ignore_empty=False):
    """ Generator which yields cutouts

    Args:
       cutout_shape: size of cutout to make, dimensions must be odd so it can be centered - [4] - (t, x, y, z)
      pos_radius: radius around annotated points that will be considered positive - int
      rotations: whether to include rotations in augmenting - bool
      partial_frac: probability that a partial cutout will be emitted - float \in [0, 1]
      sharpen_frac: probability that a 'sharpening' cutout will be emitted - float \in [0, 1]
        a 'sharpening' cutout is one that is drawn from within 2 * pos_radius of the annotation
      verbose: (optional) if True, print updates on status
      include_augment: whether to augment samples
      ignore_empty: if True, only return cutouts that have something in them

    """

    # verify shape
    assert len(cutout_shape) == 4
    # [t, x, y, z]
    offsets = []
    for cutout_dim in cutout_shape:
        # must be odd
        assert cutout_dim % 2 == 1
        offsets.append(int((cutout_dim - 1) / 2))
    spatial_offsets = offsets[1:]

    # load annotations
    full_annotations = fetch_full_annotations()

    annotated_tps = np.unique(full_annotations[:, 0]).astype(np.int32)

    if not _partials_ready(cutout_shape):
        if verbose:
            print("Did not find saved cutouts for cutout shape: {}, saving now ".format(cutout_shape))

        save_partial_cutouts(cutout_shape, offsets)

    if not _test_regions_ready():
        if verbose:
            print("Did not find saved test bounding boxes, saving now")

        set_test_regions()

    with h5py.File(STACK_PATH, 'r') as stack_file, h5py.File(PARTIAL_CUTOUTS_PATH, 'r') as partials_file:
        # [t, z, y, x]
        vol_shape = stack_file[str(annotated_tps[0])].shape[1:]

        cutout_str = str(tuple(cutout_shape))
        partial_cutouts = partials_file['{}/cutouts'.format(cutout_str)]
        partial_labels = partials_file['{}/labels'.format(cutout_str)]

        if verbose:
            print("Building division KD-trees")

        # one kd tree for every used timepoint
        nearest_divs = {}
        for t_idx in annotated_tps:
            t_annotations = full_annotations[full_annotations[:, 0] == t_idx][:, 1:]
            nearest_divs[t_idx] = NearestNeighbors(n_neighbors=10).fit(t_annotations)

        if verbose:
            print("Entering main loop")

        # begin cutout generation loop
        while True:
            roll = np.random.random()
            # yield a partial cutout
            if roll < partial_frac:
                if verbose:
                    print("Yielding chunk from partial annotations")

                # random.choice does not support N-d arrays
                idx_roll = np.random.randint(partial_cutouts.shape[0])
                cutout_roll = partial_cutouts[idx_roll]
                is_division = partial_labels[idx_roll]

                if include_augment:
                    yield is_division, augment_vol(cutout_roll)
                else:
                    yield is_division, cutout_roll

            # TODO: handle test/train

            # yield a cutout from one of the fully annotation volumes
            else:
                raise NotImplementedError()
                t_roll = np.random.choice(annotated_tps)
                roll = np.random.random()
                # yield a sharpening cutout
                if roll < sharpen_frac:
                    if verbose:
                        print("Yielding sharpening chunk")

                    # [t, z, y, x]
                    coord_roll = _draw_coord(full_annotations, t_roll)
                    # sharpen_coords, is_division = _choose_sharpen_coords(coord_roll, pos_radius, nearest_divs[t_roll], spatial_offsets[::-1], vol_shape)

                    chosen_cutout = _cutout(coord_roll, stack_file, offsets)

                    yield True, augment_vol(chosen_cutout, include_aniso=include_augment,
                                               rotations=rotations, verbose=verbose)

                # yield a cutout chosen uniformly at random, spatiotemporally
                else:
                    if verbose:
                        print("Yielding fair chunk")

                    # loop until we find a chunk satisfying our requirements
                    while True:
                        coord_roll = [np.random.randint(dim - 4 * offset) + 2 * offset for dim, offset in zip(vol_shape, spatial_offsets[::-1])]

                        full_coord_roll = [t_roll] + coord_roll

                        nbr_distances, _ = nearest_divs[t_roll].kneighbors(np.asarray(coord_roll[::-1]).reshape(1, -1))
                        is_division = (nbr_distances < pos_radius).any()
                        chosen_cutout = _cutout(full_coord_roll, stack_file, offsets)

                        if not ignore_empty or np.sum(chosen_cutout) > 0.01:
                            yield is_division, augment_vol(chosen_cutout, include_aniso=include_augment,
                                                       rotations=rotations, verbose=verbose)
                            break


def _partials_ready(cutout_shape):
    """ Returns true if cutouts have been saved for cutout_shape
    """
    try:
        with h5py.File(PARTIAL_CUTOUTS_PATH, 'r') as partials_file:
            cutout_str = str(tuple(cutout_shape))
            return cutout_str in partials_file
    except OSError:
        return False

def _test_regions_ready():
    """ Returns true if test bounding boxes have been specified
    """
    # load annotations
    full_annotations = fetch_full_annotations()
    annotated_tps = np.unique(full_annotations[:, 0]).astype(np.int32)
    with h5py.File(STACK_PATH, 'r') as stack_file:
        for t_idx in annotated_tps:
            if '{}_test_bb'.format(t_idx) not in stack_file:
                return False
        return True


def _draw_coord(annotations, t_roll):
    """ Helper method that draws an annotation fairly from all annotations at t_roll
    Also makes sure that the axis order is the right way around

    Args:
      annotations: full annotations - [n_annotations, 4], row format is [t, x, y, z]
      t_roll: draw of t - int, must have annotations at that time

    Returns:
      coord_roll: fairly drawn coord - [t, z, y, x]
    """
    # [t, x, y, z]
    t_annotations = annotations[annotations[:, 0] == t_roll]
    idx_roll = np.random.randint(len(t_annotations))
    coord_roll = t_annotations[idx_roll].tolist()
    # [t, z, y, x]
    coord_out = [coord_roll[0]] + coord_roll[1:][::-1]
    return np.rint(coord_out).astype(int)




def _pt_in_center_half(ax_len):
    """ Helper method that returns a point chosen
    uniformly at random from the interval [ax_len / 4, 3 * (ax_len / 4)]
    """
    qt_len = int(ax_len / 4)
    return np.random.randint(qt_len, 3 * qt_len)

def save_partial_cutouts(cutout_shape):
    """ Saves cutouts in an hdf5 file

    Save file is structured as:
      (cutout_shape) -> (
                 partial_cutout dataset with that shape
                 labels dataset
    )

    """
    offsets = [int((ax_len - 1) / 2) for ax_len in cutout_shape]
    partial_annotations = fetch_partial_annotations()
    ordered_shape = [cutout_shape[0]] + cutout_shape[1:][::-1]
    ordered_shape = tuple(ordered_shape)

    cutout_accum = []
    label_accum = []
    # [T, L, X, Y, Z]
    for annotation in partial_annotations:
        try:
            label, cutout = _cutout_from_disk(annotation, offsets)
            if cutout.shape == ordered_shape:
                cutout_accum.append(cutout)
                label_accum.append(label)
            else:
                print("Annotation {} failed due to shape mismatch".format(annotation))
        except ValueError as val_err:
            print("Annotation {} failed with: {}".format(annotation, val_err))
        except KeyError as key_err:
            print("Annotation {} failed with: {}".format(annotation, key_err))
        except OSError as os_err:
            print("Annotation {} failed with: {}".format(annotation, os_err))

    print("Extracted cutouts: {}/{}".format(len(cutout_accum), len(partial_annotations)))


    labels = np.stack(label_accum)
    cutouts = np.stack(cutout_accum)

    with h5py.File(PARTIAL_CUTOUTS_PATH, 'a') as partials_file:
        cutout_shape = tuple(cutout_shape)

        partials_file.create_dataset('{}/cutouts'.format(cutout_shape), data=cutouts)
        partials_file.create_dataset('{}/labels'.format(cutout_shape), data=labels)


def save_partial_splits(cutout_shape, n_pos=300, n_neg=700):
    """ Splits existing partial cutouts into train and test sets

    Args:
      cutout_shape: [t, x, y, z]
      n_pos: approximate number of positive samples to include in test
      n_neg: approximate number of negative samples to include in test
    """
    offsets = [int((ax_len - 1) / 2) for ax_len in cutout_shape]

    full_annotations = fetch_full_annotations().astype(int)
    full_tps = np.unique(full_annotations[:, 0])
    valid_annotations = fetch_validation_annotations().astype(int)
    valid_tps = np.unique(valid_annotations[:, 0])
    partial_annotations = fetch_partial_annotations().astype(int)

    n_pos_samples = len([antn for antn in partial_annotations if antn[1] in POS_LABELS])
    n_neg_samples = len(partial_annotations) - n_pos_samples

    pos_rate = n_pos / n_pos_samples
    neg_rate = n_neg / n_neg_samples

    ordered_shape = [cutout_shape[0]] + cutout_shape[1:][::-1]
    ordered_shape = tuple(ordered_shape)

    train_cutout_accum = []
    train_label_accum = []
    test_cutout_accum = []
    test_label_accum = []

    # [T, L, X, Y, Z]
    for annotation in partial_annotations:
        try:
            label, cutout = _cutout_from_disk(annotation, offsets)
            if cutout.shape == ordered_shape:
                # timepoint used as training
                if annotation[0] in full_tps:
                    train_cutout_accum.append(cutout)
                    train_label_accum.append(label)
                # timepoint used as validation
                elif annotation[0] in valid_tps:
                    test_cutout_accum.append(cutout)
                    test_label_accum.append(label)
                # randomly assigned to train/test according to rates
                else:
                    roll = np.random.random()
                    # is division
                    if label:
                        # add to test
                        if roll < pos_rate:
                            test_cutout_accum.append(cutout)
                            test_label_accum.append(label)
                        # add to train
                        else:
                            train_cutout_accum.append(cutout)
                            train_label_accum.append(label)
                    else:
                        # add to test
                        if roll < neg_rate:
                            test_cutout_accum.append(cutout)
                            test_label_accum.append(label)
                        # add to train
                        else:
                            train_cutout_accum.append(cutout)
                            train_label_accum.append(label)

            else:
                print("Annotation {} failed due to shape mismatch".format(annotation))
        except ValueError as val_err:
            print("Annotation {} failed with: {}".format(annotation, val_err))
        except KeyError as key_err:
            print("Annotation {} failed with: {}".format(annotation, key_err))
        except OSError as os_err:
            print("Annotation {} failed with: {}".format(annotation, os_err))


    train_labels = np.stack(train_label_accum)
    train_cutouts = np.stack(train_cutout_accum)

    test_labels = np.stack(test_label_accum)
    test_cutouts = np.stack(test_cutout_accum)

    print("Train pos: {} neg {} tot {}".format(np.sum(train_labels), np.sum(~train_labels), len(train_labels)))
    print("Test pos: {} neg {} tot {}".format(np.sum(test_labels), np.sum(~test_labels), len(test_labels)))

    with h5py.File(SPLIT_PARTIALS_PATH_TEMPLATE.format('train'), 'a') as partials_file:
        cutout_shape = tuple(cutout_shape)

        partials_file.create_dataset('{}/cutouts'.format(cutout_shape), data=train_cutouts)
        partials_file.create_dataset('{}/labels'.format(cutout_shape), data=train_labels)

    with h5py.File(SPLIT_PARTIALS_PATH_TEMPLATE.format('test'), 'a') as partials_file:
        cutout_shape = tuple(cutout_shape)

        partials_file.create_dataset('{}/cutouts'.format(cutout_shape), data=test_cutouts)
        partials_file.create_dataset('{}/labels'.format(cutout_shape), data=test_labels)



def _cutout_from_disk(annotation, offsets):
    """ Fetches a cutout directly from disk, using the h5 representation

    Args:
      annotation: [t, l, x, y, z]
      offsets: [t, x, y, z]

    Returns:
      is_division: bool
      cutout: of shape cutout_shape, centered on annotation
    """
    assert len(offsets) == 4
    annotation = np.asarray(annotation).astype(np.int32)
    # [z, y , x]
    spatial_offsets = offsets[1:][::-1]
    coords = annotation[2:][::-1]

    print("Making cutouts for {}".format(annotation))
    # positive label definitions
    pos_labels = {1, 2, 3, 4, 5, 103}

    # load stack
    t_stack = []
    for tl_idx in range(annotation[0] - offsets[0], annotation[0] + offsets[0] + 1):
        # [z, y, x]
        vol_name = 'SPM00_TM{:0>6d}_CM00_CM01_CHN00.fusedStack.corrected.shifted'.format(tl_idx)
        with h5py.File('{}/{}.h5'.format(VOL_DIR_H5, vol_name), 'r') as vol_file:
            vol = vol_file['vol']
            t_vol = vol[coords[0] - spatial_offsets[0]: coords[0] + spatial_offsets[0] + 1,
                        coords[1] - spatial_offsets[1]: coords[1] + spatial_offsets[1] + 1,
                        coords[2] - spatial_offsets[2]: coords[2] + spatial_offsets[2] + 1]
        t_stack.append(t_vol)
    t_stack = np.stack(t_stack)
    is_division = annotation[1] in pos_labels
    return is_division, t_stack


def write_klb_as_h5(in_dir, out_dir):
    """ Writes the full time series of volumes as compressed
    hdf5 files with the same name
    """
    import pyklb
    klb_fnames = os.listdir(in_dir)

    for klb_fname in klb_fnames:
        out_path = '{}/{}.h5'.format(out_dir, klb_fname[:-4])
        in_path = '{}/{}'.format(in_dir, klb_fname)
        if not os.path.exists(out_path):
            print("Loading {}".format(in_path))
            vol = pyklb.readfull(in_path)
            print("Saving")
            with h5py.File(out_path, 'w') as out_file:
                out_file.create_dataset('vol', data=vol, compression='lzf')

def save_normalized_h5s():
    """ Saves normalized copies of each volume

    """
    from pathos.multiprocessing import Pool
    out_dir = '/nrs/turaga/bergera/division_detection/normalized_h5_timeseries'
    in_dir = '/nrs/turaga/bergera/division_detection/full_h5_timeseries'

    pool = Pool(20)
    print("Mapping...")
    pool.map(_save_normalized_helper, os.listdir(in_dir))


def _normalized_exists(f_name):
    out_dir = '/nrs/turaga/bergera/division_detection/normalized_h5_timeseries'

    try:
        with h5py.File('{}/{}'.format(out_dir, f_name), 'r') as norm_file:
            return 'vol' in norm_file
    except:
        return False

def _save_normalized_helper(f_name):
    out_dir = '/nrs/turaga/bergera/division_detection/normalized_h5_timeseries'
    in_dir = '/nrs/turaga/bergera/division_detection/full_h5_timeseries'

    if not _normalized_exists(f_name):
        print("Normalizing {}".format(f_name))
        print(" -loading")
        with h5py.File('{}/{}'.format(in_dir, f_name)) as vol_file:
            vol = vol_file['vol'][:].astype(np.float32)
        print(' -normalizing')
        vol -= np.mean(vol, keepdims=True)
        vol *= (1 / np.std(vol, keepdims=True))
        print(' -saving')
        with h5py.File('{}/{}'.format(out_dir, f_name)) as norm_vol_file:
            norm_vol_file.create_dataset('vol', data=vol, compression='lzf')


def save_vol_stacks(n_tps=7):
    """ Saves stacks for all annotated volumes as hdf5 files

    File structure is:
      t_idx -> [n_tps, z_dim, y_dim, x_dim]
    """
    assert n_tps % 2 == 1
    t_offset = int((n_tps - 1) / 2)

    stack_path = os.path.expanduser('~/data/div_detect/vol_stacks')

    # load annotations
    full_annotations = fetch_full_annotations()
    validation_annotations = fetch_validation_annotations()

    annotated_tps = np.unique(full_annotations[:, 0]).astype(np.int32)
    val_tps = np.unique(validation_annotations[:, 0]).astype(np.int32)

    with h5py.File('{}/stacks.h5'.format(stack_path)) as stack_file:
        # this translates to list(iter(stack_file))
        # there's probably a better way
        existing_tps = [int(key) for key in stack_file]
        missing_tps = (set(annotated_tps) | set(val_tps)) - set(existing_tps)
        missing_tps = np.array(list(missing_tps)).astype(np.int32)
        print("existing tps: {} \n missing tps: {}".format(existing_tps, missing_tps))

        for t_idx in missing_tps:
            print("Loading stack for t = {}".format(t_idx))
            # load stack
            t_stack = []
            for tl_idx in range(t_idx - t_offset, t_idx + t_offset + 1):
                t_stack.append(fetch_vol(tl_idx))
            t_stack = np.stack(t_stack)

            # write stack
            print("Writing stack")
            stack_file.create_dataset(str(t_idx), data=t_stack, compression='lzf')

def save_gt_render(pos_radius=3, name='gts', crosshair=False):
    """ Saves GTs as rendered volumes
    """
    save_path = os.path.expanduser('~/data/div_detect/gt_vols')

    # load annotations
    full_annotations = fetch_full_annotations()

    annotated_tps = np.unique(full_annotations[:, 0]).astype(np.int32)

    # [z, y, x]
    gt_shape = (972, 393, 417)

    with h5py.File('{}/{}.h5'.format(save_path, name), 'w') as stack_file:
        for t_idx in annotated_tps:
            gt_vol = np.zeros(gt_shape)
            for x_idx, y_idx, z_idx in full_annotations[full_annotations[:, 0] == t_idx][:, 1:]:
                gt_coord = (int(z_idx), int(y_idx / 5) , int(x_idx / 5))
                if crosshair:
                    insert_crosshair(gt_vol, gt_coord, crosshair_radius=pos_radius)
                else:
                    insert_ball(gt_vol, gt_coord, radius=pos_radius)

            # write stack
            print("Writing stack")
            stack_file.create_dataset(str(t_idx), data=gt_vol)


def save_full_res_gt_render(pos_radius=2, name='train_target'):
    save_path = os.path.expanduser('~/data/div_detect/full_res_gt_vols')

    # [z, y, x]
    vol_shape = fetch_vol_shape()

    # load annotations
    full_annotations = fetch_full_annotations()

    annotated_tps = np.unique(full_annotations[:, 0]).astype(np.int32)

    # [z, y, x]
    gt_shape = vol_shape

    with h5py.File('{}/{}.h5'.format(save_path, name), 'w') as stack_file:
        for t_idx in annotated_tps:
            gt_vol = np.zeros(gt_shape)
            for x_idx, y_idx, z_idx in full_annotations[full_annotations[:, 0] == t_idx][:, 1:]:
                gt_coord = (int(z_idx), int(y_idx) , int(x_idx))
                insert_ball(gt_vol, gt_coord, radius=pos_radius)

            # write stack
            print("Writing stack")
            stack_file.create_dataset(str(t_idx), data=gt_vol, compression='lzf')

def save_full_res_validation_gt_renders(pos_radius=5, name='validation'):
    save_path = os.path.expanduser('~/data/div_detect/full_res_gt_vols')

    # [z, y, x]
    vol_shape = fetch_vol_shape()

    # load annotations
    full_annotations = fetch_validation_annotations()

    annotated_tps = np.unique(full_annotations[:, 0]).astype(np.int32)

    # [z, y, x]
    gt_shape = vol_shape

    with h5py.File('{}/{}.h5'.format(save_path, name), 'w') as stack_file:
        for t_idx in annotated_tps:
            gt_vol = np.zeros(gt_shape)
            for x_idx, y_idx, z_idx in full_annotations[full_annotations[:, 0] == t_idx][:, 1:]:
                gt_coord = (int(z_idx), int(y_idx) , int(x_idx))
                insert_ball(gt_vol, gt_coord, radius=pos_radius)

            # write stack
            print("Writing stack")
            stack_file.create_dataset(str(t_idx), data=gt_vol, compression='lzf')



def save_partial_gt_renders(pos_radius=5, name='partial_render'):
    save_path = os.path.expanduser('~/data/div_detect/full_res_gt_vols')

    # [z, y, x]
    vol_shape = fetch_vol_shape()

    # load annotations
    annotations = fetch_partial_annotations()

    annotated_tps = np.unique(annotations[:, 0]).astype(np.int32)

    # [z, y, x]
    gt_shape = vol_shape

    with h5py.File('{}/{}.h5'.format(save_path, name), 'w') as stack_file:
        for t_idx in annotated_tps:
            gt_vol = np.zeros(gt_shape)
            for label, x_idx, y_idx, z_idx in annotations[annotations[:, 0] == t_idx][:, 1:]:
                gt_coord = (int(z_idx), int(y_idx) , int(x_idx))
                is_division = label in POS_LABELS
                if is_division:
                    insert_ball(gt_vol, gt_coord, radius=pos_radius)
                else:
                    insert_crosshair(gt_vol, gt_coord, crosshair_radius=pos_radius)

            # write stack
            print("Writing stack")
            stack_file.create_dataset(str(t_idx), data=gt_vol, compression='lzf')




def insert_ball(arr, coord, radius=5):
    """ Adds a ball centered around coord

    Args:
      arr: array with ndim == 3
      coord: center of crosshair - [x, y, z]
      radius: radius of ball - int

    Returns:
      None

    """
    from itertools import product

    # build up valid displacements by brute force over [-radius, radius]^3
    # only okay because dim = 3. as dim -> infinity the fraction of integer coordinates
    #  satisfying (magnitude <= radius) to the total number of coordinates goes to zero.....
    ball_displacements = []
    for trial_coord in product(range(-radius, radius + 1), repeat=3):
        if np.sum(np.array(trial_coord) ** 2) <= radius ** 2:
            ball_displacements.append(trial_coord)

    for displacement in ball_displacements:
        try:
            arr[coord[0] + displacement[0],
                coord[1] + displacement[1],
                coord[2] + displacement[2]] = 1
        except IndexError:
            pass



def insert_crosshair(arr, coord, crosshair_radius=3):
    """ Adds a 'crosshair' at  coord

    Args:
      arr: array with ndim == 3
      coord: center of crosshair - [x, y, z]
      crosshair_radius: specifies how wide to make the crosshair - int

    Returns:
      None
    """

    # x axis
    for x_idx in range(coord[0] - crosshair_radius, coord[0] + crosshair_radius + 1):
        try:
            arr[x_idx, coord[1], coord[2]] = 1
        except IndexError:
            pass

    # y axis
    for y_idx in range(coord[1] - crosshair_radius, coord[1] + crosshair_radius + 1):
        try:
            arr[coord[0], y_idx, coord[2]] = 1
        except IndexError:
            pass

    # z axis
    for z_idx in range(coord[2] - crosshair_radius, coord[2] + crosshair_radius + 1):
        try:
            arr[coord[0], coord[1], z_idx] = 1
        except IndexError:
            pass



def _choose_sharpen_coords(coord_roll, pos_radius, neighbors, spatial_offsets, vol_shape):
    """ Get a random sharpening coordinate, offset from coord_roll

    Args:
      coord_roll: division location - [t, z, y, x]
      pos_radius: radius around coord_roll which we consider dividing - float
      neighbors: fit NearestNeighbors instance, giving knn divs
      spatial_offsets: define receptive field size - [z, y, x]
      vol_shape: [z, y, x]

    Returns:
      sharpening_coord: randomly offset from coord_roll, within 2 * pos_radius
      is_division: label for coord - bool
    """
    assert len(vol_shape) == 3
    assert len(spatial_offsets) == 3
    spatial_coords = coord_roll[1:]


    # now draw radial offset in range [0, 2 * pos_rad]
    radial_offset = np.random.random() * 2 * pos_radius
    is_division = radial_offset < pos_radius

    # draw angles
    theta_roll = np.random.random() * np.pi
    phi_roll = np.random.random() * np.pi * 2

    # get cartesian displacements
    x_displacement = radial_offset * np.sin(theta_roll) * np.cos(phi_roll)
    y_displacement = radial_offset * np.sin(theta_roll) * np.sin(phi_roll)
    # anisotropy correction
    z_displacement = radial_offset * np.cos(theta_roll) / 5
    displacement = np.array([z_displacement, y_displacement, x_displacement])
    trial_coord = spatial_coords + displacement

    # positive labels are always ok no matter their proximity to neighbors
    # it's only negative we have to check
    if not is_division:
        nbr_distances, _ = neighbors.kneighbors(trial_coord.reshape(1, -1))
        nz_nbr_distances = nbr_distances[nbr_distances >  0]
        if (nz_nbr_distances <  pos_radius).any():
            # recurse until we have a valid coord
            # this is safe because the positive case occurs 50% of the time
            #   and is always accepted
            return _choose_sharpen_coords(coord_roll, pos_radius, neighbors, spatial_offsets, vol_shape)

    for ax_comp, ax_len, ax_offset in zip(trial_coord, vol_shape[1:], spatial_offsets):
        edge_dist = np.min([ax_comp, ax_len - ax_comp])
        assert edge_dist >= 0
        # if window is out of bounds, redraw coord
        if edge_dist < ax_offset:
            return _choose_sharpen_coords(coord_roll, pos_radius, neighbors, spatial_offsets, vol_shape)

    trial_coord = trial_coord.tolist()
    chosen_coord = [coord_roll[0]] + trial_coord
    return chosen_coord, is_division


def _cutout(coords, vol_file, offsets):
    """ Cutout helper method

    Args:
      coords: t, z, y, x
      vol_file: h5py file with the stacks as datasets
      offsets: offsets in t, x, y, z

    Returns:
     cutout: vol - [t, z, y, x]
    """
    assert len(coords) == 4
    assert len(offsets) == 4

    coords = np.asarray(coords).astype(np.int32)
    offsets = np.asarray(offsets).astype(np.int32)

    # [z, y , x]
    spatial_offsets = offsets[1:][::-1]

    # [t, x, y, z]
    vol_stack = vol_file[str(int(coords[0]))]

    # assert that the saved vols are wide enough in time
    assert vol_stack.shape[0] >= offsets[0] * 2 + 1
    t_mdpt = int((vol_stack.shape[0] - 1) /2)

    return vol_stack[t_mdpt - offsets[0]: t_mdpt + offsets[0] + 1,
                     coords[1] - spatial_offsets[0]: coords[1] + spatial_offsets[0] + 1,
                     coords[2] - spatial_offsets[1]: coords[2] + spatial_offsets[1] + 1,
                     coords[3] - spatial_offsets[2]: coords[3] + spatial_offsets[2] + 1]



def augment_vol(cutout, include_aniso=False, roll=None):
    """

    Args:
      cutout: [t, z, y, x]
      include_aniso: if True, include the rotations that require resampling
      roll: optional roll in range [0, 16] - float

    """

    if roll is not None:
        assert 0 <= roll <= 8
    else:
        roll = np.random.random() * 8

    if include_aniso:
        raise NotImplementedError()
    else:
        return fair_d4_action(cutout, roll=roll)



def _downsample(cube):
    """
    Args:
      cube: array - [t, z, y, x]
    """
    # list element shapes are [t, 5, y, x]
    pre_ds_split = np.split(cube, cube.shape[-1] /  5, axis=1)
    ds_split = [z_split.mean(axis=1) for z_split in pre_ds_split]
    return np.stack(ds_split, axis=1)


def fair_d4_action(cutout, roll=None):
    """ Returns a fair random D4 action on the cutout, applied about the z axis

    Args:
      cutout: [t, z, x, y]
      roll: optional roll in range [0, 8] - float

    Returns:
       cutout_augment: rotated/reflected cutout

    """
    assert cutout.ndim == 4

    # uniform float \in [0, 8]
    # range is for convenience, there are 8 rotations and reflections
    if roll is not None:
        assert 0 <= roll <= 8
    else:
        roll = np.random.random() * 8

    # reflect
    if roll >= 4:
        cutout = cutout[:, ::-1, :, :]
        roll -= 4

    rot_idx = int(math.floor(roll))
    return rot90(cutout, rot_idx, (2, 3))

def cube_isotopy(cube, roll=None):
    """ Returns a fair random isotopy of the input cube

    Makes use of external code credited in preprocessing
    """
    cube = cube.squeeze()
    assert cube.ndim == 3
    assert cube.shape[0] == cube.shape[1] == cube.shape[2]

    # uniform float \in [0, 47]
    # range is for convenience, there are 48 rotations and reflections
    if roll is not None:
        assert 0 <= roll <= 47
    else:
        roll = np.random.random() * 47

    # reflect
    if roll >= 24:
        cube = cube[::-1, ::-1, ::-1]
        roll -= 24

    # roll range is now [0, 23]
    # range [0, 6]
    sub_sel_roll = roll / 4.

    if sub_sel_roll < 1:
        rot_idx = int(math.floor(roll))
        return rot90(cube, rot_idx, 0)
    if 1 <= sub_sel_roll < 2:
        cube = rot90(cube, 2, axis=1)
        rot_idx = int(math.floor(roll - 4))
        return rot90(cube, rot_idx, 0)
    if 2 <= sub_sel_roll < 3:
        cube = rot90(cube, 1, axis=1)
        rot_idx = int(math.floor(roll - 8))
        return rot90(cube, rot_idx, 2)
    if 3 <= sub_sel_roll < 4:
        cube = rot90(cube, -1, axis=1)
        rot_idx = int(math.floor(roll - 12))
        return rot90(cube, rot_idx, 2)
    if 4 <= sub_sel_roll < 5:
        cube = rot90(cube, 1, axis=2)
        rot_idx = int(math.floor(roll - 16))
        return rot90(cube, rot_idx, 1)
    else:
        cube = rot90(cube, -1, axis=2)
        rot_idx = int(math.floor(roll - 18))
        return rot90(cube, rot_idx, 1)

def fetch_vol_shape():
    """ Fetches the vol shape, for a single timepoint
    """
    vol_name = 'SPM00_TM{:0>6d}_CM00_CM01_CHN00.fusedStack.corrected.shifted'.format(100)
    with h5py.File('{}/{}.h5'.format(VOL_DIR_H5, vol_name)) as vol_file:
       return vol_file['vol'].shape


def fetch_stack(t_pred):
    """ Fetches the vols for predicting t_idx
    """
    vol_shape = fetch_vol_shape()
    t_stack = np.zeros([7] + list(vol_shape))
    for idx, t_idx in enumerate(range(t_pred - 3, t_pred + 4)):
        t_stack[idx] = fetch_vol(t_idx)
    return t_stack

def fetch_stack_cutout(t_pred, cutout_coords):
    """ Fetches a cutout from a full slice without loading the whole slice first

    Args:
      t_pred: timepoint to predict on, midpoint in t returned
      cutout_coords: [(z_min, z_max), (y_min, y_max), (z_min, z_max)]
    """
    assert len(cutout_coords) == 3
    cutout_shape = [ax_bb[1] - ax_bb[0] for ax_bb in cutout_coords]
    t_stack = np.zeros([7] + cutout_shape)
    for idx, t_idx in enumerate(range(t_pred - 3, t_pred + 4)):
        t_stack[idx] = fetch_vol_cutout(t_idx, cutout_coords)
    return t_stack

def fetch_vol_cutout(t_idx, cutout_coords):
    """ Fetches cutout from volume at t_idx

    Args:
      t_idx: timepoint to fetch
      cutout_coords: [(z_min, z_max), (y_min, y_max), (z_min, z_max)]
    """
    vol_name = 'SPM00_TM{:0>6d}_CM00_CM01_CHN00.fusedStack.corrected.shifted'.format(t_idx)
    with h5py.File('{}/{}.h5'.format(VOL_DIR_H5, vol_name)) as vol_file:
       return vol_file['vol'][cutout_coords[0][0]:cutout_coords[0][1],
                              cutout_coords[1][0]:cutout_coords[1][1],
                              cutout_coords[2][0]:cutout_coords[2][1]]


def fetch_vol(t_idx, as_h5=True):
    """ Fetches the vol for at the timepoint t_idx

    """

    vol_name = 'SPM00_TM{:0>6d}_CM00_CM01_CHN00.fusedStack.corrected.shifted'.format(t_idx)
    if as_h5:
        with h5py.File('{}/{}.h5'.format(VOL_DIR_H5, vol_name)) as vol_file:
           return vol_file['vol'][:]
    else:
        import pyklb
        warn("loading raw klbs; not normalized")
        return pyklb.readfull('{}/{}.klb'.format(VOL_DIR, vol_name))

def fetch_partial_annotations():
    """ Returns the partial annotations as an array

    Returns:
      partial_annotations: array of annotation data - [n_annotations, 5]
        row format is [T, L, X, Y, Z]
    """
    raw_mat = loadmat(PARTIAL_ANNOTATIONS_PATH)
    annotations = raw_mat['divisionAnnotations']
    # chop extra mystery column
    return annotations[:, :-1]


def _annotation_generator():
    """ Helper generator over all full volume annotations
    """
    # all full annotations
    with open(FULL_ANNOTATIONS_PATH, 'r') as annot_file:
        for line_idx, line in enumerate(annot_file):
            if line_idx >= 1:
                cast_line = [float(tok) for tok in line.split(',')[:4]]
                yield cast_line

    with open(VALIDATION_ANNOTATIONS_PATH, 'r') as annot_file:
        for line in annot_file:
            # [t, x, y, z]
            cast_line = [float(tok) for tok in line.split(',')[:4]]
            yield cast_line


def fetch_all_annotations():
    """ Returns all annotations

    Returns:
    complete_annotations: array of annotation data - [n_annotations, 4]
        row format is [T, X, Y, Z]

    """
    ann_gen = _annotation_generator()

    data = []
    for annotation in ann_gen:
        data.append(annotation)

    data = np.asarray(data)
    # scale z down to expected range
    data *= [1, 1, 1, 0.2]
    return data


def fetch_train_annotations():
    """ Returns the annotations earmarked for training

    Returns:
    complete_annotations: array of annotation data - [n_annotations, 4]
        row format is [T, X, Y, Z]

    """
    ann_gen = _annotation_generator()
    val_test_tps = VAL_TIMEPOINTS + TEST_TIMEPOINTS

    data = []
    for annotation in ann_gen:
        if annotation[0] not in val_test_tps:
            data.append(annotation)

    data = np.asarray(data)
    # scale z down to expected range
    data *= [1, 1, 1, 0.2]
    return data


def fetch_validation_annotations():
    """ Returns the validation annotations

    Returns:
    complete_annotations: array of annotation data - [n_annotations, 4]
        row format is [T, X, Y, Z]

    """
    ann_gen = _annotation_generator()

    data = []
    for annotation in ann_gen:
        if annotation[0] in VAL_TIMEPOINTS:
            data.append(annotation)
    data = np.asarray(data)
    # scale z down to expected range
    data *= [1, 1, 1, 0.2]
    return data

def fetch_test_annotations():
    """ Returns the test set annotations

    Returns:
    complete_annotations: array of annotation data - [n_annotations, 4]
        row format is [T, X, Y, Z]

    """
    ann_gen = _annotation_generator()

    data = []
    for annotation in ann_gen:
        if annotation[0] in TEST_TIMEPOINTS:
            data.append(annotation)
    data = np.asarray(data)
    # scale z down to expected range
    data *= [1, 1, 1, 0.2]
    return data

def fetch_corrected_annotations(timepoint, label_level=1):
    """ Returns the corrected annotations

    Args:
      timepoint: tp to fetch the corrections for
      label_level: yields rows with label levels less than this - int
          see below description

    Returns:
      annotations: [n_annotations, 6]
         row format is [X, Y, Z, T, Score, Label]
          or sometimes [X, Y, Z, T, Score, Label, T division, centered?]


           Label format is:
             0 = No Division
             1 = Division (within +/-2 time points relative to center time point)
             2 = Division (beyond +/-2 time points relative to center time point or relatively far from spatial center)
             3 = Cannot make a definitive decision


    """
    col_types = [int, int, int, int, float, int, int, str]
    # some timepoints have extra columns with annotations on where divisions occurred
    if timepoint in [120, 240, 360]:
        n_cols = 8
    else:
        n_cols = 6

    annotations = []

    annot_path = '{}/{}.csv'.format(CORRECTED_ANNOTATIONS_PATH, timepoint)
    with open(annot_path, 'r') as annot_file:
        for line_idx, line in enumerate(annot_file):
            # skip header
            if line_idx > 1:
                row = []
                split_line = line.split(',')[1:n_cols + 1]
                for token, tok_type in zip(split_line, col_types):
                    if len(token) > 0:
                        row.append(tok_type(token))
                    else:
                        if tok_type in [int, float]:
                            row.append(np.nan)
                        else:
                            row.append('')
                if row[5] <= label_level:
                    annotations.append(row)
    annotations = np.asarray(annotations)
    return annotations


def fetch_annotations(mode):
    """ Fetch annotations by mode

    Args:
       mode: one of ['train', 'test', 'validation']

    Returns:
      annotations: array of annotation data - [n_annotations, 4]
        row format is [T, X, Y, Z]
    """
    if mode == 'train':
        return fetch_train_annotations()
    elif mode == 'validation':
        return fetch_validation_annotations()
    elif mode == 'test':
        return fetch_test_annotations()
    else:
        raise NotImplementedError("Unrecognized mode: {}".format(mode))



def n_chunks(chunk_size, overlap=(16, 84, 84)):
    """ Convenience method that returns the number of chunks for these parameters

    """
    vol_name_template = 'SPM00_TM{:0>6d}_CM00_CM01_CHN00.fusedStack.corrected.shifted.h5'
    with h5py.File('{}/{}'.format(VOL_DIR_H5, vol_name_template.format(100))) as vol_file:
        vol_shape = vol_file['vol'].shape

    n_chunks_by_dim = [ceil(vol_ax_len / float(chunk_ax_len - ax_overlap)) for vol_ax_len, chunk_ax_len, ax_overlap in zip(vol_shape, chunk_size, overlap)]
    return np.prod(n_chunks_by_dim)


def streaming_regular_chunker(chunk_size, overlap=(16, 84, 84)):
    """ Very much like regular regular chunker except that this one has time in the inner loop
    instead of the outer loop and manually enforces the convolutional efficiency along the time axis
    the spatial dimensions are actually convolutions

    Generator over regular chunks of chunk size centered around t_predict
    overlap should be receptive field size - 1

    z, y, x

    """
    raise NotImplementedError()
    vol_name_template = 'SPM00_TM{:0>6d}_CM00_CM01_CHN00.fusedStack.corrected.shifted.h5'

    with h5py.File('{}/{}'.format(VOL_DIR_H5, vol_name_template.format(100))) as vol_file:
        vol_shape = vol_file['vol'].shape

    n_chunks_by_dim = [ceil(vol_ax_len / float(chunk_ax_len - ax_overlap)) for vol_ax_len, chunk_ax_len, ax_overlap in zip(vol_shape, chunk_size, overlap)]


    try:
        vol_timeseries = [h5py.File('{}/{}'.format(VOL_DIR_H5, vol_name)) for vol_name in sorted(os.listdir(VOL_DIR_H5))]

        for z_idx in range(n_chunks_by_dim[0]):
            z_coord = z_idx * (chunk_size[0] - overlap[0])
            if z_coord + chunk_size[0] >= vol_shape[0] and n_chunks_by_dim[0] > 1:
                z_coord = vol_shape[0] - chunk_size[0]
            for y_idx in range(n_chunks_by_dim[1]):
                y_coord = y_idx * (chunk_size[1] - overlap[1])
                if y_coord + chunk_size[1] >= vol_shape[1] and n_chunks_by_dim[1] > 1:
                    y_coord = vol_shape[1] - chunk_size[1]
                for x_idx in range(n_chunks_by_dim[2]):
                    x_coord = x_idx * (chunk_size[2] - overlap[2])
                    if x_coord + chunk_size[2] >= vol_shape[2] and n_chunks_by_dim[2] > 1:
                        x_coord = vol_shape[2] - chunk_size[2]

                    for t_predict in range(3, len(vol_timeseries) - 4):
                    # load stack
                        chunk = np.empty([7] + list(chunk_size))
                        for tl_idx in range(7):
                            # [z, y, x]
                            vol = vol_timeseries[tl_idx + t_predict - 3]['vol']
                            chunk[tl_idx] = vol[z_coord: z_coord + chunk_size[0],
                                                y_coord: y_coord + chunk_size[1],
                                                x_coord: x_coord + chunk_size[2]]

                        chunk_coords = (t_predict, z_coord, y_coord, x_coord)
                        yield chunk, chunk_coords
    except:
        for vol_file in vol_timeseries:
            vol_file.close()
        raise

def streaming_regular_chunker_time_last(chunk_size, overlap=(16, 84, 84)):
    """ Very much like regular regular chunker except that this one has time in the outer loop

    Generator over regular chunks of chunk size centered around t_predict
    overlap should be receptive field size - 1

    z, y, x

    """
    raise NotImplementedError()
    vol_name_template = 'SPM00_TM{:0>6d}_CM00_CM01_CHN00.fusedStack.corrected.shifted.h5'

    with h5py.File('{}/{}'.format(VOL_DIR_H5, vol_name_template.format(100))) as vol_file:
        vol_shape = vol_file['vol'].shape

    n_chunks_by_dim = [ceil(vol_ax_len / float(chunk_ax_len - ax_overlap)) for vol_ax_len, chunk_ax_len, ax_overlap in zip(vol_shape, chunk_size, overlap)]


    try:
        vol_timeseries = [h5py.File('{}/{}'.format(VOL_DIR_H5, vol_name)) for vol_name in sorted(os.listdir(VOL_DIR_H5))]

        for t_predict in range(3, len(vol_timeseries) - 4):
            for z_idx in range(n_chunks_by_dim[0]):
                z_coord = z_idx * (chunk_size[0] - overlap[0])
                if z_coord + chunk_size[0] >= vol_shape[0] and n_chunks_by_dim[0] > 1:
                    z_coord = vol_shape[0] - chunk_size[0]
                for y_idx in range(n_chunks_by_dim[1]):
                    y_coord = y_idx * (chunk_size[1] - overlap[1])
                    if y_coord + chunk_size[1] >= vol_shape[1] and n_chunks_by_dim[1] > 1:
                        y_coord = vol_shape[1] - chunk_size[1]
                    for x_idx in range(n_chunks_by_dim[2]):
                        x_coord = x_idx * (chunk_size[2] - overlap[2])
                        if x_coord + chunk_size[2] >= vol_shape[2] and n_chunks_by_dim[2] > 1:
                            x_coord = vol_shape[2] - chunk_size[2]

                        # load stack
                        chunk = np.empty([7] + list(chunk_size))
                        for tl_idx in range(7):
                            # [z, y, x]
                            vol = vol_timeseries[tl_idx + t_predict - 3]['vol']
                            chunk[tl_idx] = vol[z_coord: z_coord + chunk_size[0],
                                                y_coord: y_coord + chunk_size[1],
                                                x_coord: x_coord + chunk_size[2]]

                        chunk_coords = (t_predict, z_coord, y_coord, x_coord)
                        yield chunk, chunk_coords
    except:
        for vol_file in vol_timeseries:
            vol_file.close()
        raise


def in_mem_chunker(t_predict, chunk_size, padding=[4, 22, 22]):
    """ Chunker that first loads the entire stack into memory

    Args:
      t_predict: timepoint to predict on
      chunk_size: biggest chunk that will fit on gpu - [z, y, x]
      padding: one-sided output padding - [z, y, x]
    """
    print("Fetching stack for {}".format(t_predict))
    t_stack = fetch_stack(t_predict)
    ax_scaling = [1, 1, 1]

    output_chunk_size = [int(ax_size / ax_scale) - 2 * ax_pad for ax_size, ax_scale, ax_pad in zip(chunk_size, ax_scaling, padding)]

    vol_name_template = 'SPM00_TM{:0>6d}_CM00_CM01_CHN00.fusedStack.corrected.shifted.h5'

    with h5py.File('{}/{}'.format(VOL_DIR_H5, vol_name_template.format(t_predict))) as vol_file:
        vol_shape = vol_file['vol'].shape

    n_chunks_by_dim = [ceil(vol_ax_len / float(chunk_ax_len)) for vol_ax_len, chunk_ax_len in zip(vol_shape, output_chunk_size)]


    print("Yielding chunks")
    for z_idx in range(n_chunks_by_dim[0]):
        z_coord = z_idx * output_chunk_size[0] * ax_scaling[0]
        if z_coord + chunk_size[0] >= vol_shape[0] and n_chunks_by_dim[0] > 1:
            z_coord = vol_shape[0] - chunk_size[0]
        for y_idx in range(n_chunks_by_dim[1]):
            y_coord = y_idx * output_chunk_size[1] * ax_scaling[1]
            if y_coord + chunk_size[1] >= vol_shape[1] and n_chunks_by_dim[1] > 1:
                y_coord = vol_shape[1] - chunk_size[1]
            for x_idx in range(n_chunks_by_dim[2]):
                x_coord = x_idx * output_chunk_size[2] * ax_scaling[2]
                if x_coord + chunk_size[2] >= vol_shape[2] and n_chunks_by_dim[2] > 1:
                    x_coord = vol_shape[2] - chunk_size[2]

                chunk = t_stack[:, z_coord: z_coord + chunk_size[0],
                                y_coord: y_coord + chunk_size[1],
                                x_coord: x_coord + chunk_size[2]]

                chunk_coords = (z_coord, int(y_coord / ax_scaling[1]), int(x_coord/ ax_scaling[2]))
                yield chunk, chunk_coords



def regular_chunker(t_predict, chunk_size, padding=(4, 22, 22)):
    """ Generator over regular chunks of chunk size centered around t_predict
    overlap should be receptive field size - 1

    z, y, x

    """
    vol_name_template = VOL_DIR_H5 + '/SPM00_TM{:0>6d}_CM00_CM01_CHN00.fusedStack.corrected.shifted.h5'
    t_stack = [h5py.File(vol_name_template.format(t_idx), 'r')['vol'] for t_idx in range(t_predict - 3, t_predict + 4)]

    bbox = fetch_bb_at_t(t_predict)

    ax_scaling = [1, 1, 1]

    output_chunk_size = [int(ax_size / ax_scale) - 2 * ax_pad for ax_size, ax_scale, ax_pad in zip(chunk_size, ax_scaling, padding)]

    vol_name_template = 'SPM00_TM{:0>6d}_CM00_CM01_CHN00.fusedStack.corrected.shifted.h5'

    with h5py.File('{}/{}'.format(VOL_DIR_H5, vol_name_template.format(t_predict))) as vol_file:
        vol_shape = vol_file['vol'].shape

    n_chunks_by_dim = [ceil(vol_ax_len / float(chunk_ax_len)) for vol_ax_len, chunk_ax_len in zip(vol_shape, output_chunk_size)]

    try:
        print("Yielding chunks")
        for z_idx in range(n_chunks_by_dim[0]):
            z_coord = z_idx * output_chunk_size[0] * ax_scaling[0]
            if z_coord + chunk_size[0] >= vol_shape[0] and n_chunks_by_dim[0] > 1:
                z_coord = vol_shape[0] - chunk_size[0]
            for y_idx in range(n_chunks_by_dim[1]):
                y_coord = y_idx * output_chunk_size[1] * ax_scaling[1]
                if y_coord + chunk_size[1] >= vol_shape[1] and n_chunks_by_dim[1] > 1:
                    y_coord = vol_shape[1] - chunk_size[1]
                for x_idx in range(n_chunks_by_dim[2]):
                    x_coord = x_idx * output_chunk_size[2] * ax_scaling[2]
                    if x_coord + chunk_size[2] >= vol_shape[2] and n_chunks_by_dim[2] > 1:
                        x_coord = vol_shape[2] - chunk_size[2]

                    chunk_coord = (z_coord, y_coord, x_coord)
                    if _chunk_in_bb(chunk_coord, chunk_size, bbox):
                        chunk = np.zeros([7] + list(chunk_size))
                        for c_idx in range(7):
                            chunk[c_idx] = t_stack[c_idx][z_coord: z_coord + chunk_size[0],
                                                          y_coord: y_coord + chunk_size[1],
                                                          x_coord: x_coord + chunk_size[2]]

                        chunk_coords = (z_coord, int(y_coord / ax_scaling[1]), int(x_coord/ ax_scaling[2]))
                        yield chunk, chunk_coords
    except Exception as general_err:
        print("Caught exception: {}".format(general_err))
    # Make sure we always close open files
    finally:
        for vol in t_stack:
            vol.file.close()


def general_regular_chunker(t_predict, process_dir, chunk_size, padding=(4, 22, 22)):
    """ Regular chunker that handles the general prediction format
    """
    # check dir structure
    assert process_dir.endswith('.h5')

    assert t_predict < 100, "Change vol_name_template in division_detection.vol_preprocessing.general_regular_chunker for support of larger t_predict values"

    logger = logging.getLogger("division_detection.chunker")

    vol_name_template = 'Volume_{:0>2d}.h5'
    vol_path_template = process_dir + '/' + vol_name_template
    vol_name = vol_name_template.format(t_predict)
    t_stack = [h5py.File(vol_path_template.format(t_idx), 'r')['vol'] for t_idx in range(t_predict - 3, t_predict + 4)]
    vol_shape = t_stack[0].shape

    with h5py.File('{}/{}'.format(process_dir, vol_name), 'r') as volume:
        bbox = volume['bbox']
        bbox = [
            (bbox[4], bbox[5]),
            (bbox[2], bbox[3]),
            (bbox[0], bbox[1])
        ]

    ax_scaling = [1, 1, 1]

    output_chunk_size = [int(ax_size / ax_scale) - 2 * ax_pad for ax_size, ax_scale, ax_pad in zip(chunk_size, ax_scaling, padding)]
    logger.info("Output chunk size: %s", output_chunk_size)

    n_chunks_by_dim = [int(ceil(vol_ax_len / float(chunk_ax_len))) for vol_ax_len, chunk_ax_len in zip(vol_shape, output_chunk_size)]
    logger.info("Number of chunks per dimension: %s", n_chunks_by_dim)

    try:
        logger.info("Yielding chunks")
        for z_idx in range(n_chunks_by_dim[0]):
            z_coord = z_idx * output_chunk_size[0] * ax_scaling[0]
            if z_coord + chunk_size[0] >= vol_shape[0] and n_chunks_by_dim[0] > 1:
                z_coord = vol_shape[0] - chunk_size[0]
            for y_idx in range(n_chunks_by_dim[1]):
                y_coord = y_idx * output_chunk_size[1] * ax_scaling[1]
                if y_coord + chunk_size[1] >= vol_shape[1] and n_chunks_by_dim[1] > 1:
                    y_coord = vol_shape[1] - chunk_size[1]
                for x_idx in range(n_chunks_by_dim[2]):
                    x_coord = x_idx * output_chunk_size[2] * ax_scaling[2]
                    if x_coord + chunk_size[2] >= vol_shape[2] and n_chunks_by_dim[2] > 1:
                        x_coord = vol_shape[2] - chunk_size[2]

                    chunk_coord = (z_coord, y_coord, x_coord)
                    if _chunk_in_bb(chunk_coord, chunk_size, bbox):
                        chunk = np.zeros([7] + list(chunk_size))
                        for c_idx in range(7):
                            chunk[c_idx] = t_stack[c_idx][z_coord: z_coord + chunk_size[0],
                                                          y_coord: y_coord + chunk_size[1],
                                                          x_coord: x_coord + chunk_size[2]]

                        chunk_coords = (z_coord, int(y_coord / ax_scaling[1]), int(x_coord/ ax_scaling[2]))
                        yield chunk, chunk_coords
    except Exception as general_err:
        logger.critical("Caught exception: %s", general_err)
        raise
    # Make sure we always close open files
    finally:
        for vol in t_stack:
            vol.file.close()


def fetch_callbacks(model_spec):
    """ Fetch standard callbacks for div detection pipeline

    Args:
      model_spec: as returned by model.generate_model_spec

    Returns:
       list of keras callbacks
    """
    train_spec = model_spec['train_spec']
    save_dir = '/groups/turaga/home/bergera/results/div_detect/model/mk2'
    model_dir = '{}/{}'.format(save_dir, model_spec['name'])
    tb_log_dir = '/groups/turaga/home/bergera/logs/division_detection/tb_logs/{}'.format(model_spec['name'])

    if os.path.exists(model_dir):
        warn("Found existing save files, overriding")
        for file_name in os.listdir(model_dir):
            os.remove('{}/{}'.format(model_dir, file_name))
    else:
        os.mkdir(model_dir)


    if os.path.exists(tb_log_dir):
        warn('Found existing log dir under same name, overriding')
        for file_name in os.listdir(tb_log_dir):
            os.remove('{}/{}'.format(tb_log_dir, file_name))
    else:
        print("Creating log dir")
        os.mkdir(tb_log_dir)

    callbacks = [
        # stop training if val loss does not improve for 50 epochs
        EarlyStopping(patience=train_spec['early_stopping_patience'],
                      verbose=1
        ),
        # cuts learning rate when val loss stagnates
        ReduceLROnPlateau(factor=0.5,
                          patience=train_spec['lr_plateau_patience'],
                          cooldown=0,
                          verbose=1
        ),
        # computes histograms of layer activations every 10 epochs
        TensorBoard(log_dir=tb_log_dir,
                    write_graph=False,
                    histogram_freq=5
        ),
        ModelCheckpoint(model_dir + '/best_weights.{epoch:02d}-{val_loss:.5f}.h5',
                        verbose=1,
                        period=1,
                        save_weights_only=True,
                        save_best_only=False)
    ]
    return callbacks

def bbox_nd(img, thresh=1e-4):
    """ External code from http://stackoverflow.com/questions/31400769/bounding-box-of-numpy-array
    """
    from itertools import combinations
    N = img.ndim
    out = []
    for ax in combinations(range(N), N - 1):
        nonzero = np.any(img > thresh, axis=ax)
        out.extend(np.where(nonzero)[0][[0, -1]])
    return [int(k) for k in out]


def bb_from_annotations(t_idx, padding=100):
    """ Compute a rough bounding box from annotations plus some margin

    Args:
      t_idx: timepoint to compute for. must have annotations
      padding: padding width in xy units
    """
    # load annotations
    # [t, x, y, z]
    full_annotations = fetch_full_annotations()
    annotated_tps = np.unique(full_annotations[:, 0]).astype(np.int32)

    assert t_idx in annotated_tps

    # [z, y, x]
    annotations_at_t = full_annotations[full_annotations[:, 0] == t_idx][:, 1:][:, ::-1]

    padding_scale = [0.2, 1, 1]

    # [(z_min, z_max), (y_min, y_max), (x_min, x_max)]
    bbox = [(np.min(annotations_at_t[:, idx]), np.max(annotations_at_t[:, idx])) for idx in range(3)]
    bbox = [(int(bb[0] - padding * scale), int(bb[1] + padding * scale)) for bb, scale in zip(bbox, padding_scale)]
    return bbox

def fetch_bb_at_t(t_idx):
    """ Convenience method, loads the bounding box at timepoint t_idx from disk

    Args:
      t_idx: timepoint - int

    Returns
      bbox: [(z_min, z_max), (y_min, y_max), (z_min, z_max)]
    """
    with open(BBOXES_PATH, 'r') as bbox_file:
        bbox = json.load(bbox_file)[str(int(t_idx))]
        formatted_bbox = [
            (bbox[4], bbox[5]),
            (bbox[2], bbox[3]),
            (bbox[0], bbox[1])
        ]
        return formatted_bbox


def fetch_bboxes():
    """ Convenience method, fetches bounding boxes for all annotated tps

    Returns:
      bboxes: {time_pt: bbox}
    """
    raise NotImplementedError()

def save_bboxes(thresh=300, max_tp=None):
    """ Calculate bounding boxes for all volumes and save as a json
    """
    import json
    from pathos.multiprocessing import Pool
    vol_path = '/nrs/turaga/bergera/division_detection/full_h5_timeseries'
    vol_template = vol_path + '/SPM00_TM{:0>6d}_CM00_CM01_CHN00.fusedStack.corrected.shifted.h5'
    save_path = os.path.expanduser('~/data/div_detect/bboxes/bboxes.json')
    n_tps = len(os.listdir(vol_path))
    n_tps =  max_tp or n_tps

    tps = np.arange(n_tps)

    pool = Pool(40)

    def _get_bbox(t_idx):
        with h5py.File(vol_template.format(t_idx), 'r') as vol_file:
            return bbox_nd(vol_file['vol'][:], thresh=thresh)


    bbox_list = pool.map(_get_bbox, tps)
    bboxes = {int(t_idx): bbox for t_idx,  bbox in zip(tps, bbox_list)}

    with open(save_path, 'w') as bbox_file:
        json.dump(bboxes, bbox_file)

def save_bboxes_general(in_dir, thresh=300):
    """ More general version that computes bboxes for all files in in_dir
    and writes their bboxes to the h5 file
    """
    logger = logging.getLogger('division_detection.{}'.format(__name__))

    def _do_bbox(fpath):
        with h5py.File(fpath, 'r+') as volume:
            if 'bbox' not in volume:
                bbox = bbox_nd(volume['vol'][:], thresh=thresh)
                volume['bbox'] = volume.create_dataset('bbox', bbox)
                return True
            return False



    fnames = os.listdir(in_dir)
    h5_fpaths = ['{}/{}'.format(in_dir, fname) for fname in  fnames]

    computed = map(_do_bbox, h5_fpaths)
    logger.info("Computed bounding boxes for %s volumes", sum(computed))

def continuous_augment(vol, rotation, flip_z):
    """ Apply a continuous augmentation to vol

    Args:
      vol: [t, z, y, x] or [z, y, x]
      rotation: radians, in range [0, 2 * pi]
      flip_z: if True, flip z axis - bool

    Returns:
      augmented_vol: [t, z, y, x] or [z, y, x]
    """
    assert 0 < rotation < 2 * math.pi

    if vol.ndim == 4:
        vol_shape = vol.shape[1:]
    else:
        vol_shape = vol.shape

    transformation = augment.create_identity_transformation(vol_shape)
    transformation += augment.create_rotation_transformation(vol_shape, rotation)

    if flip_z:
        if vol.ndim == 4:
            vol = vol[:, ::-1]
        else:
            vol =  vol[::-1]

    if vol.ndim == 4:
        vol_slices = np.split(vol, 7, axis=0)
        aug_slices = [augment.apply_transformation(vol_slice.squeeze(), transformation) for vol_slice in vol_slices]
        return np.stack(aug_slices, axis=0)
    else:
        return augment.apply_transformation(vol, transformation)


def continuous_augment_shape_transform(cutout_shape):
    """ Transforms the cutout shape into that required for the continuous augmentation super-selection

    Args:
      cutout_shape: [z, y, x] where y == x

    Returns:
      supersel_shape: [z_s, y_s, x_s]

    """
    assert len(cutout_shape) == 3
    assert cutout_shape[1] == cutout_shape[2]

    supersel_shape = np.asarray((cutout_shape[0], cutout_shape[1] * np.sqrt(2), cutout_shape[2] * np.sqrt(2)))
    return supersel_shape.astype(int)

def continuous_augment_coord_transform(corner_coord, cutout_len):
    """ Transforms the corner coord into that required for the continuous augmentation super-selection

    Args:
      corner_coord: lower corner coord of *output* [z, y, x]
      cutout_len: cutout x/y axis len

    """
    corner_coord = np.asarray(corner_coord)
    offset = continuous_augment_offset(cutout_len)
    vec_offset = np.array([0, -offset, -offset])
    return (corner_coord + vec_offset).astype(int)

def continuous_augment_offset(cutout_len):
    """ Return the cartesian offset for the continuous augmentation super-selection

    Args:
      cutout_len: cutout x/y axis len

    Returns:
      offset: cartesian offset - float
    """
    offset = ((np.sqrt(2) - 1) / 2) * cutout_len
    return int(offset)

# =================================
#      PRIVATE METHODS
# =================================

def _chunk_in_bb(chunk_corner_coord, chunk_shape, bbox):
    """ Returns True if any part of the chunk is contained in the bounding box
    """
    assert len(chunk_corner_coord) == len(chunk_shape) == len(bbox)
    for ax_coord, ax_shape, ax_bb in zip(chunk_corner_coord,
                                         chunk_shape,
                                         bbox):
        if _interval_in_bb(ax_coord, ax_shape, ax_bb):
            return True
    return False

def _interval_in_bb(interval_left_coord, interval_width, bbox):
    """ 1D chunk_in_bb

    """
    assert len(bbox) == 2
    bbox_lo, bbox_hi = bbox
    interval_right_coord = interval_left_coord + interval_width
    if bbox_lo < interval_left_coord < bbox_hi:
        return True
    elif bbox_lo < interval_right_coord < bbox_hi:
        return True
    # only remaining case is bbox fully contained in interval
    elif ((interval_left_coord < bbox_lo < interval_right_coord) and
          (interval_left_coord < bbox_hi < interval_right_coord)):
        return True
    else:
        return False

# =================================
#             DEPRECATED
# =================================

def fetch_full_annotations():
    """ Returns the complete annotations as an array
    Annotations can be assumed to cover all positive divisions for a given time

    Returns:
    complete_annotations: array of annotation data - [n_annotations, 4]
        row format is [T, X, Y, Z]

    """
    raise DeprecationWarning("Use fetch_train_annotation or fetch_all_annotations")

def set_test_regions(test_region_shape=[100, 500, 500]):
    """ Randomly picks test regions within the fully annotated volumes to be reserved for testing

    Test regions are saved as a dataset of bounding boxes named {tp}_test_bb

    Args:
      test_region_shape: shape of region to reserve for testing - [z, y, x], in anisotropic units

    """
    # load annotations
    full_annotations = fetch_full_annotations()

    annotated_tps = np.unique(full_annotations[:, 0]).astype(np.int32)

    test_volume = np.prod(test_region_shape)

    with h5py.File(STACK_PATH, 'a') as stack_file:
        # [z, y, x]
        full_vol_shape = stack_file[str(annotated_tps[0])].shape[1:]
        full_volume = np.prod(full_vol_shape)

        test_frac = test_volume / full_volume

        print("Test fraction: {} ".format(test_frac))

        # report test fraction
        # draw and save regions from within center half

        for t_idx in annotated_tps:
            inner_corner_coord = [_pt_in_center_half(ax_len) for ax_len in full_vol_shape]
            bounding_box = [(coord, coord + width) for coord, width in zip(inner_corner_coord,
                                                                           test_region_shape)]
            bounding_box = np.asarray(bounding_box)

            dset_name = '{}_test_bb'.format(t_idx)

            stack_file.create_dataset(dset_name, data=bounding_box)
