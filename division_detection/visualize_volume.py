""" Script that uses neuroglancer to visualize a volume
"""
import numpy as np

import neuroglancer
import h5py
import os
import sys
from division_detection.utils import vol_to_int8


def run_vol(vol_idx):
    neuroglancer.set_static_content_source(url='https://neuroglancer-demo.appspot.com')

    with h5py.File(os.path.expanduser('~/data/div_detect/full_records.h5'), 'r') as rec_file:
        record = rec_file['records'][vol_idx,:, :, :, 1]

    viewer = neuroglancer.Viewer(voxel_size=[1, 1, 1])
    viewer.add(vol_to_int8(record),
               name='record')
    #viewer.add(b, name='b')
    return viewer

if __name__ == 'main':
    # argument should be index of volume to visualize
    viewer = run_vol(int(sys.argv[1]))
    print(viewer)
