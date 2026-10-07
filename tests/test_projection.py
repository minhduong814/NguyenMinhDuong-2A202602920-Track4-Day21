import unittest

import numpy as np

from starter.kitti_io import KittiCalib
from starter.projection import cam_to_image, velo_to_cam


class ProjectionTests(unittest.TestCase):
    def setUp(self):
        self.identity_calib = KittiCalib(
            P2=np.array([[10, 0, 5, 0], [0, 10, 5, 0], [0, 0, 1, 0]], dtype=float),
            R0_rect=np.eye(3),
            Tr_velo_to_cam=np.column_stack([np.eye(3), np.zeros(3)]),
        )

    def test_velo_to_cam_applies_homogeneous_transform(self):
        calib = self.identity_calib
        calib.Tr_velo_to_cam[:, 3] = [1, 2, 3]
        actual = velo_to_cam(np.array([[1, 1, 1]], dtype=float), calib)
        np.testing.assert_allclose(actual, [[2, 3, 4]])

    def test_cam_to_image_projects_and_returns_source_mask(self):
        points = np.array([[0, 0, 2], [20, 0, 2], [0, 0, -1], [np.nan, 0, 2]], dtype=float)
        uv, depth, mask = cam_to_image(points, self.identity_calib.P2, (10, 10, 3))
        np.testing.assert_allclose(uv, [[5, 5]])
        np.testing.assert_allclose(depth, [2])
        np.testing.assert_array_equal(mask, [True, False, False, False])

    def test_cam_to_image_empty_input(self):
        uv, depth, mask = cam_to_image(np.empty((0, 3)), self.identity_calib.P2, (10, 10, 3))
        self.assertEqual(uv.shape, (0, 2))
        self.assertEqual(depth.shape, (0,))
        self.assertEqual(mask.shape, (0,))


if __name__ == "__main__":
    unittest.main()
