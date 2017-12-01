""" This module contains utils for working with sparse Nd arrays
"""

import numpy as np
import h5py


def dense_nd_to_coo(arr_nd, thresh=1e-5):
    """ Convert a dense Nd array into coo format

    Args:
      arr_nd: array like
      thresh: cutoff values, values below thresh will be discarded - float

    Returns:
      coo_arr: rows have format [ax_0_idx, ax_1_idx, ..., ax_n_idx, val] - [n_nz, ndim + 1]
             indices take on dtype of data and must be cast back to int for use
    """
    arr_nd = np.asarray(arr_nd)
    sparse_sel = list(np.where(arr_nd > thresh))
    # [n_nz, ndim + 1] - [ax_0_idx, ax_1_idx, ..., ax_n_idx, val]
    coo_repr = np.stack(sparse_sel + [arr_nd[sparse_sel]]).T.astype(arr_nd.dtype)
    return coo_repr

def coo_to_dense_nd(coo_repr, shape):
    """  Convert a sparse array in COO format into a dense array

    Args:
      coo_repr: coo format sparse representation - [n_nz, ndim + 1]
      shape: shape of dense array - [ndim]

    Returns:
      dense_arr: array in original shape
    """
    ndim = coo_repr.shape[1] - 1
    assert len(shape) == ndim

    arr_nd = np.zeros(shape, dtype=coo_repr.dtype)
    arr_nd[tuple(coo_repr.T[:-1].astype(int))] = coo_repr.T[-1]
    return arr_nd

def save_dense_as_coo(arr_nd, filename, thresh=1e-5):
    """ Converts to coo and saves as a h5

    Args:
      arr_nd: array like
      filename: absolute path to desired save location - str
      thresh: cutoff values, values below thresh will be discarded - float

    """
    if not filename.endswith('.h5') or filename.endswith('.hdf5'):
        filename += '.h5'

    coo_repr = dense_nd_to_coo(arr_nd, thresh=thresh)

    with h5py.File(filename, 'w') as coo_file:
        coo_file.create_dataset('coo', data=coo_repr)
        coo_file.create_dataset('shape', data=arr_nd.shape)

def load_coo_as_dense(filename):
    """ Loads a coo file as a dense array

    Args:
       filename: absolute path to file location - str

    Returns:
      dense_arr: array in original shape

    """
    if not filename.endswith('.h5') or filename.endswith('.hdf5'):
        filename += '.h5'

    with h5py.File(filename, 'r') as coo_file:
        assert 'shape' in coo_file
        assert 'coo' in coo_file
        return coo_to_dense_nd(coo_file['coo'][:], coo_file['shape'][:])

def load_coo(filename):
    """ Loads a coo file in coo format

    Args:
       filename: absolute path to file location - str

    Returns:
      coo_arr: rows have format [ax_0_idx, ax_1_idx, ..., ax_n_idx, val] - [n_nz, ndim + 1]
         indices take on dtype of data and must be cast back to int for use
      shape: int array - [ndim]

    """
    if not filename.endswith('.h5') or filename.endswith('.hdf5'):
        filename += '.h5'

    with h5py.File(filename, 'r') as coo_file:
        assert 'shape' in coo_file
        assert 'coo' in coo_file
        return coo_file['coo'][:], coo_file['shape'][:]
