import unittest

import numpy as np

from src.autolabel import bbox_iou, points_in_box
from starter.kitti_io import KittiObject


class AutoLabelGeometryTests(unittest.TestCase):
    def setUp(self):
        self.obj = KittiObject(
            type="Car",
            truncated=0.0,
            occluded=0,
            alpha=0.0,
            bbox=np.array([0.0, 0.0, 10.0, 10.0]),
            dimensions=np.array([2.0, 2.0, 4.0]),
            location=np.array([0.0, 1.0, 10.0]),
            rotation_y=0.0,
        )

    def test_bbox_iou(self):
        self.assertAlmostEqual(bbox_iou(np.array([0, 0, 10, 10]), np.array([5, 0, 15, 10])), 1 / 3)
        self.assertEqual(bbox_iou(None, np.array([0, 0, 1, 1])), 0.0)

    def test_points_in_axis_aligned_box(self):
        points = np.array([[0, 0, 10], [1.9, 1, 10], [0, -1, 10], [0, 0, 11.1]])
        np.testing.assert_array_equal(points_in_box(points, self.obj, tolerance_m=0.0), [True, True, True, False])

    def test_points_in_rotated_box(self):
        rotated = KittiObject(**{**self.obj.__dict__, "rotation_y": np.pi / 2})
        points = np.array([[0.9, 0, 11.9], [1.1, 0, 11.9]])
        np.testing.assert_array_equal(points_in_box(points, rotated, tolerance_m=0.0), [True, False])


if __name__ == "__main__":
    unittest.main()
