from __future__ import annotations

import unittest

import cv2
import numpy as np

from backend.geometry import order_quad, rectify_polygon


class GeometryTests(unittest.TestCase):
    def test_order_quad_is_stable_for_unordered_points(self) -> None:
        points = np.array([[90, 70], [10, 10], [12, 80], [100, 12]], dtype=np.float32)
        ordered = order_quad(points)
        np.testing.assert_array_equal(ordered[0], [10, 10])
        np.testing.assert_array_equal(ordered[1], [100, 12])
        np.testing.assert_array_equal(ordered[2], [90, 70])
        np.testing.assert_array_equal(ordered[3], [12, 80])

    def test_rectification_returns_a_large_clean_crop(self) -> None:
        image = np.full((500, 800, 3), 235, dtype=np.uint8)
        polygon = np.array([[120, 90], [700, 55], [735, 410], [85, 440]], dtype=np.float32)
        cv2.polylines(image, [polygon.astype(np.int32)], True, (20, 30, 20), 6)
        result = rectify_polygon(image, polygon)
        self.assertGreaterEqual(max(result.image.shape[:2]), 1190)
        self.assertGreater(result.image.shape[1], result.image.shape[0])
        self.assertEqual(result.source_quad.shape, (4, 2))
        self.assertEqual(result.transform.shape, (3, 3))


if __name__ == "__main__":
    unittest.main()

