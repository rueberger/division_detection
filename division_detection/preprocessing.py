"""  This module contains methods for data preprocessing
"""
import itertools
import os
import h5py
import numpy as np
import random
from scipy.ndimage.interpolation import zoom

from glob import glob

from division_detection.utils import round_arr_up
from scipy.io import loadmat

base_dir = '/nrs/turaga/bergera/division_detection'
label_file = 'annotations/divisionAnnotations.mat'
stack_dir = 'stacks_4d'

# total records in dataset
N_RECORDS = 2926
# 48 distinct rotations and reflections of a cube
N_ISOTOPIES = 48

CUTOUT_SHAPE = [21, 105, 105]

def preprocess_vol(vol):
    """ Forces isotropic and downsamples

    Args:
      vol: vol to process - np.ndarray

    Returns:
      ds_vol: isotropic and downsampled volume


    """
    zoom_factor = 0.185
    if vol.ndim == 4:
        ds_vol = zoom(vol, (zoom_factor * 5, zoom_factor, zoom_factor, 1), order=1)
    elif vol.ndim == 3:
        ds_vol = zoom(vol, (zoom_factor * 5, zoom_factor, zoom_factor), order=1)
    else:
        raise NotImplementedError

    return ds_vol




def to_isotropic(anisotropic_vol, z_ax=0):
    """ Convert an anisotropic volume to an equivalent isotropic one

    Args:
      anisotropic_vol: ndarray, relative axis sizes of 5 x 1 x 1  - [z_dim, y_dim, x_dim]

    Returns:
      isotropic_vol: ndarray - [z_dim * 5, y_dim, x_dim]
    """
    return np.repeat(anisotropic_vol, 5, axis=z_ax)

def to_cube(isotropic_vol):
    """  Cubifies isotropic_vol
    Centered and zero-padded

    Args:
      isotropic_vol: ndarray - [x_dim, y_dim, z_dim]

    returns:
      cubed_vol: ndarray - [max(x_dim, y_dim, z_dim)] * 3
    """
    max_dim = np.max(isotropic_vol.shape)
    ax_padding = []
    for ax_l in isotropic_vol.shape:
        l_pad = max_dim - int(ax_l / 2)
        r_pad = max_dim  - l_pad - ax_l
    ax_padding.append((l_pad, r_pad))
    return np.lib.pad(isotropic_vol, ax_padding, mode='constant', constant_values=0)


def oh_action(vol):
    """ Generate all equivalent volumes
    Formally, this is the image of the group action of
    the octahedral group O_h \sim S_3 \times C_2 on the voxel
    There are 48 distinct elements

    Args:
      vol: isotropic, cubed vol  - [x_dim] * 3

    Return:
      oh_img: list of all distinct elements of the group action - [np.ndarray]

    """
    assert len(np.unique(vol.shape)) == 1
    flipped_vol = vol[::-1, ::-1, ::-1]
    orbit = [rot_vol for rot_vol in rotations24(vol)]
    flipped_orbit = [rot_vol for rot_vol in rotations24(flipped_vol)]
    orbit.extend(flipped_orbit)
    return orbit

def data_reader_generator():
    """ Returns a generator over data and labels
    """
    # 1 codes for division
    label_code = {
        1: 1,
        2: 1,
        3: 1,
        4: 1,
        5: 1,
        103: 1,
        0: 0,
        100: 0}
    label_name = {
        1: 'A',
        2: 'B',
        3: 'C',
        4: 'D',
        5: 'E',
        0: 'W',
        103: 'CT',
        100: 'WT'
    }
    label_path = "{}/{}".format(base_dir, label_file)
    # [n_labeled, 6]
    annotations = loadmat(label_path)['divisionAnnotations']
    for label_idx in range(annotations.shape[0]):
        annotation = annotations[label_idx]
        # format is
        # time label x y z ?
        time_pt = annotation[0]
        division_label = label_code[annotation[1]]
        xyz_coords = round_arr_up(annotation[2:5])
        # format for the stack names is 4 digit zero padded int from left
        stack_name = 'StackTimeSeries.Class_{}.TM_{:0>4d}.Centroid_{:0>4d}_{:0>4d}_{:0>4d}.klb'.format(
            label_name[annotation[1]],
            int(annotation[0]),
            xyz_coords[0],
            xyz_coords[1],
            xyz_coords[2]
        )

        vol_path = '{}/{}/{}'.format(base_dir, stack_dir, stack_name)
        # [t, z, y, x] = [7, 21, 81, 81]
        vol_timeseries = pyklb.readfull(vol_path)
        yield division_label, vol_timeseries, vol_path

def all_data_gen():
    """ Generator over all data sources
    """
    for record in data_reader_generator():
        yield record
    # TODO: iterate over cutout generator


def process_and_save(t_receptive_field=3):
    """ Reads from disk, generates fiber set

    Args:
      t_receptive_field: size of receptive field in t. can be 3, 5, 7
    """
    assert t_receptive_field in [3, 5, 7]

    t_cut_idxs = {
        3: (2, 5),
        5: (1, 6),
        7: (0, 7)
    }[t_receptive_field]

    cutout_shape = CUTOUT_SHAPE
    assert cutout_shape[0] * 5 == cutout_shape[1] == cutout_shape[2]
    data_dir = os.path.expanduser('~/data/div_detect/t{}'.format(t_receptive_field))

    # TODO: iterate over saved and generate cutouts

    with h5py.File('{}/full_records.h5'.format(data_dir), 'w') as record_file:
        record_shape = (N_RECORDS * N_ISOTOPIES, cutout_shape[0], cutout_shape[1], cutout_shape[2], 3)
        rec_dset = record_file.create_dataset("records", record_shape,  dtype=float)
        label_dset = record_file.create_dataset('labels', (N_RECORDS * N_ISOTOPIES, 1), dtype=float)
        h5_str_dt = h5py.special_dtype(vlen=str)
        path_dset = record_file.create_dataset('record_paths', (N_RECORDS * N_ISOTOPIES, ), dtype=h5_str_dt)
        dset_ptr = 0
        for rec_idx, (label, vol_ts, vol_path) in enumerate(data_reader_generator()):
            print("Loading rec {}".format(rec_idx))
            # make isotropic
            # [[cutout_shape] * t_receptive_field]
            t_split = [np.squeeze(split) for split in np.split(vol_ts[t_cut_idxs[0]:t_cut_idxs[1], :, :, :], t_receptive_field, axis=0)]
            # [[cutout_shape[1]] * 3] * t_receptive_field]
            cubed_t_split = [to_isotropic(split) for split in t_split]
            # [cutout_shape[1] * 3, t_receptive_field]
            iso_vol = np.stack(cubed_t_split).T

            print("Generating isotropies for rec {}".format(rec_idx))
            # generate fiber set aka isotopies
            # [[[cutout_shape[1] * 3] * 48] * 3]
            split_fiber_sets = [oh_action(iso_vol[:, :, :, t_idx]) for t_idx in range(t_receptive_field)]

            print("Saving rec {}".format(rec_idx))
            for idx in range(48):
                # [cutout_shape[1]] * 3 + [t_receptive_field]
                curr_vol = np.stack([split_fiber_sets[t_idx][idx] for t_idx in range(t_receptive_field)], axis=-1)
                # back to anisotropic
                ds_vol = zoom(curr_vol, (0.2, 1, 1, 1), order=1)
                rec_dset[dset_ptr, :, :, :, :] = ds_vol
                label_dset[dset_ptr] = label
                path_dset[dset_ptr] = vol_path
                dset_ptr += 1

def save_holdout_split(train_frac=0.7, test_frac=0.2, val_frac=0.1):
    """ Uses different records for test and train, guaranteeing fair results
    """
    print("Generating splits")
    data_dir = os.path.expanduser('~/data/div_detect')
    split_idxs = {}

    n_holdouts = int(train_frac * N_RECORDS)
    train_idxs = np.arange(n_holdouts * N_ISOTOPIES)
    np.random.shuffle(train_idxs)
    split_idxs['train'] = train_idxs

    test_val_frac = test_frac + val_frac
    test_val_idxs = np.arange(n_holdouts * N_ISOTOPIES, N_RECORDS * N_ISOTOPIES)
    np.random.shuffle(test_val_idxs)

    # renormalize
    test_frac = test_frac / test_val_frac

    test_val_split_idx = int(test_frac *  len(test_val_idxs))

    split_idxs['test'] = test_val_idxs[:test_val_split_idx]
    split_idxs['valid'] = test_val_idxs[test_val_split_idx:]

    print("Saving splits")
    with h5py.File('{}/splits.h5'.format(data_dir), 'w') as split_file:
        for split_name, idxs in split_idxs.items():
            split_dset = split_file.create_dataset(split_name, data=idxs)

    print("Pre splitting records...")

    with h5py.File('{}/full_records.h5'.format(data_dir), 'r') as full_rec_file:
        for split_name, idxs in split_idxs.items():
            print("Now running {}".format(split_name))
            with h5py.File('{}/{}_recs.h5'.format(data_dir, split_name), 'w') as split_rec_file:
                n_recs = len(idxs)
                rec_dset = split_rec_file.create_dataset("records", (n_recs, 15, 15, 15, 3),  dtype=float)
                label_dset = split_rec_file.create_dataset('labels', (n_recs, 1), dtype=float)
                h5_str_dt = h5py.special_dtype(vlen=str)
                path_dset = split_rec_file.create_dataset('record_paths', (n_recs,), dtype=h5_str_dt)

                idxs = sorted(idxs)
                rec_dset[...] = full_rec_file['records'][idxs]
                label_dset[...] = full_rec_file['labels'][idxs]
                path_dset[...] = full_rec_file['record_paths'][idxs]



def batch_generator(split_name, batch_size=32, slideable_shape=False):
    """ A generator over batches. Loops over the split forever

    Args:
      split_name: one of test, train, valid - str
      batch_size: int

    next returns:
       batch: array - [batch_size, 15, 15, 15, 3]
       label: array - [batch_size, 2]
    """
    from division_detection.utils import take
    data_dir = os.path.expanduser('~/data/div_detect')
    record_path = '{}/{}_recs.h5'.format(data_dir, split_name)

    with h5py.File(record_path, 'r') as record_file:
        print("Preloading dataset")
        records = record_file['records'][:]
        labels = record_file['labels'][:]
    split_idxs = np.arange(len(records))

    while True:
        batch_idxs = np.random.choice(split_idxs, batch_size, replace=False)
        batch_recs = np.stack(records[batch_idxs])
        batch_labels = np.stack(labels[batch_idxs])
        # expand label shape
        if slideable_shape:
            batch_labels = batch_labels.reshape(-1, 1, 1, 1, 1)
        yield batch_recs, batch_labels

def calculate_cell_splits(split_dict, n_records):
    """ Calculate split idxs for cell splits

    Args:
      split_dict: {'test': test_frac, 'train': train_frac, 'valid': valid_frac} - dict
      n_records: number of cells to split - int



    Returns:
      split_idxs: {'test': [test_cell_idxs], 'train': [train_cell_idxs] 'valid': [valid_cell_idxs]}

    """
    import math
    assert math.isclose(np.sum(list(split_dict.values())), 1)

    shuffled_idxs = np.arange(n_records)
    np.random.shuffle(shuffled_idxs)

    # may not sum to n_records initially
    trial_idxs = {key: int(n_records * split_frac) for key, split_frac in split_dict.items()}
    trial_sum = np.sum(list(trial_idxs.values()))

    if (np.asarray(trial_idxs.values()) == 0).any():
        raise RuntimeError("Cannot make zero split. Increase split frac or number of cells")

    # make the test set larger
    if trial_sum < n_records:
        trial_idxs['test'] += n_records - trial_sum
    # make the train set smaller
    elif trial_sum > n_records:
        trial_idxs['train'] -= trial_sum  - n_records

    assert np.sum(list(trial_idxs.values())) == n_records

    split_idxs = {}
    cum_idx = 0
    for key, idx_count in trial_idxs.items():
        split_idxs[key] = shuffled_idxs[cum_idx: cum_idx + idx_count]
        cum_idx += idx_count

    assert np.sum([len(set_idxs) for set_idxs in split_idxs.values()]) == n_records
    return split_idxs


def extract_all_cutouts():
    """ Extract postive and negative cutouts from a labeled volumes
    """
    label_pattern  = '/nrs/turaga/bergera/division_detection/labeled_vols/labels/*_labels.csv'

    for label_path in glob(label_pattern):
        # xyzt
        labels = np.loadtxt(label_path, delimiter=',')[:, :-1]
        # label files should have the format tp_{}_labels.csv where {} is the tp
        t_pt = int(label_path.split('/')[-1].split('_')[1])
        print("Extracting labels for time point {}".format(t_pt))
        extract_cutouts_helper(labels, t_pt)




def extract_cutouts_helper(labels, t_pt):
    """ Extract and save the cutouts
    """
    cutout_dir = '/nrs/turaga/bergera/division_detection/cutouts'
    vol_dir = '/nrs/turaga/bergera/division_detection/labeled_vols/vols'
    vol_name_template = 'SPM00_TM{:0>6d}_CM00_CM01_CHN00.fusedStack.corrected.shifted.klb'

    cutout_offset = [(ax_size - 1) / 2 for ax_size in CUTOUT_SHAPE]


    # [[z, y, x] * t]
    full_vols = []
    # fetch 7 timepoints centered around t_pt
    for t_idx in range(t_pt - 3, t_pt + 2):
        full_vols.append(pyklb.readfull(vol_dir + vol_name_template.format(t_idx)))

    # same order as keller cutouts come in
    # [t, z, y, x]
    full_vols = np.stack(full_vols, axis=0)

    # make sure we have at least one cutout per positive label
    for idx, cell_coords in enumerate(labels):
        # rounded to nearest int
        cell_coords = round_arr_up(cell_coords)
        cutout_slices = [slice(centroid - offset, centroid + offset) for centroid, offset in zip(cell_coords, cutout_offset)]
        cutout = full_vols[:, cutout_slices[0], cutout_slices[1], cutout_slices[2]]
        # TODO: write cutouts




# =============  external code from http://stackoverflow.com/questions/33190042/how-to-calculate-all-24-rotations-of-3d-array ============

def rotations24(polycube):
    # imagine shape is pointing in axis 0 (up)

    # 4 rotations about axis 0
    for rot in rotations4(polycube, 0):
        yield rot

    # rotate 180 about axis 1, now shape is pointing down in axis 0
    # 4 rotations about axis 0
    for rot in rotations4(rot90(polycube, 2, axis=1), 0):
        yield rot

    # rotate 90 or 270 about axis 1, now shape is pointing in axis 2
    # 8 rotations about axis 2
    for rot in rotations4(rot90(polycube, axis=1), 2):
        yield rot
    for rot in rotations4(rot90(polycube, -1, axis=1), 2):
        yield rot

    # rotate about axis 2, now shape is pointing in axis 1
    # 8 rotations about axis 1
    for rot in rotations4(rot90(polycube, axis=2), 1):
        yield rot
    for rot in rotations4(rot90(polycube, -1, axis=2), 1):
        yield rot

def rotations4(polycube, axis):
    """List the four rotations of the given cube about the given axis."""
    for i in range(4):
        yield rot90(polycube, i, axis)


def rot90(m, k=1, axes=(0,1)):
    """
    Rotate an array by 90 degrees in the plane specified by axes.
    Rotation direction is from the first towards the second axis.
    .. versionadded:: 1.12.0
    Parameters
    ----------
    m : array_like
        Array of two or more dimensions.
    k : integer
        Number of times the array is rotated by 90 degrees.
    axes: (2,) array_like
        The array is rotated in the plane defined by the axes.
        Axes must be different.
    Returns
    -------
    y : ndarray
        A rotated view of `m`.
    See Also
    --------
    flip : Reverse the order of elements in an array along the given axis.
    fliplr : Flip an array horizontally.
    flipud : Flip an array vertically.
    Notes
    -----
    rot90(m, k=1, axes=(1,0)) is the reverse of rot90(m, k=1, axes=(0,1))
    rot90(m, k=1, axes=(1,0)) is equivalent to rot90(m, k=-1, axes=(0,1))
    Examples
    --------
    >>> m = np.array([[1,2],[3,4]], int)
    >>> m
    array([[1, 2],
           [3, 4]])
    >>> np.rot90(m)
    array([[2, 4],
           [1, 3]])
    >>> np.rot90(m, 2)
    array([[4, 3],
           [2, 1]])
    >>> m = np.arange(8).reshape((2,2,2))
    >>> np.rot90(m, 1, (1,2))
    array([[[1, 3],
            [0, 2]],
          [[5, 7],
           [4, 6]]])
    """
    axes = tuple(axes)
    if len(axes) != 2:
        raise ValueError("len(axes) must be 2.")

    m = np.asanyarray(m)

    if axes[0] == axes[1] or np.absolute(axes[0] - axes[1]) == m.ndim:
        raise ValueError("Axes must be different.")

    if (axes[0] >= m.ndim or axes[0] < -m.ndim
        or axes[1] >= m.ndim or axes[1] < -m.ndim):
        raise ValueError("Axes={} out of range for array of ndim={}."
            .format(axes, m.ndim))

    k %= 4

    if k == 0:
        return m[:]
    if k == 2:
        return flip(flip(m, axes[0]), axes[1])

    axes_list = np.arange(0, m.ndim)
    axes_list[axes[0]], axes_list[axes[1]] = axes_list[axes[1]], axes_list[axes[0]]

    if k == 1:
        return np.transpose(flip(m,axes[1]), axes_list)
    else:
        # k == 3
        return flip(np.transpose(m, axes_list), axes[1])
    return m

def flip(m, axis):
    """
    Reverse the order of elements in an array along the given axis.
    The shape of the array is preserved, but the elements are reordered.
    .. versionadded:: 1.12.0
    Parameters
    ----------
    m : array_like
        Input array.
    axis : integer
        Axis in array, which entries are reversed.
    Returns
    -------
    out : array_like
        A view of `m` with the entries of axis reversed.  Since a view is
        returned, this operation is done in constant time.
    See Also
    --------
    flipud : Flip an array vertically (axis=0).
    fliplr : Flip an array horizontally (axis=1).
    Notes
    -----
    flip(m, 0) is equivalent to flipud(m).
    flip(m, 1) is equivalent to fliplr(m).
    flip(m, n) corresponds to ``m[...,::-1,...]`` with ``::-1`` at position n.
    Examples
    --------
    >>> A = np.arange(8).reshape((2,2,2))
    >>> A
    array([[[0, 1],
            [2, 3]],
           [[4, 5],
            [6, 7]]])
    >>> flip(A, 0)
    array([[[4, 5],
            [6, 7]],
           [[0, 1],
            [2, 3]]])
    >>> flip(A, 1)
    array([[[2, 3],
            [0, 1]],
           [[6, 7],
            [4, 5]]])
    >>> A = np.random.randn(3,4,5)
    >>> np.all(flip(A,2) == A[:,:,::-1,...])
    True
    """
    if not hasattr(m, 'ndim'):
        m = np.asarray(m)
    indexer = [slice(None)] * m.ndim
    try:
        indexer[axis] = slice(None, None, -1)
    except IndexError:
        raise ValueError("axis=%i is invalid for the %i-dimensional input array"
                         % (axis, m.ndim))
    return m[tuple(indexer)]
