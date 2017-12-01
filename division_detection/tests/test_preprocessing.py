""" Unit tests for preprocessing
"""

import unittest
import numpy as np
from division_detection.preprocessing import oh_action

class TestPreprocessing(unittest.TestCase):

    def test_symmetries(self):
        """ Check that the 48 symmetries are distinct
        """
        test_vol = np.arange(27).reshape((3, 3, 3))
        eq_class = oh_action(test_vol)
        hashes = []
        for eq_vol in eq_class:
            eq_vol.flags.writeable = False
            hashes.append(hash(eq_vol.data.tobytes()))
        self.assertTrue(len(set(hashes)),
                        msg='Not all equivalency class members distinct')
