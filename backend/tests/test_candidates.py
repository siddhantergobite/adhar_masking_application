import unittest

import cv2
import numpy as np

from backend.candidates import OpenCvDocumentProposer


class CandidateProposalTests(unittest.TestCase):
    def test_large_quadrilateral_is_only_a_geometric_proposal(self):
        image = np.full((800, 1000, 3), 72, dtype=np.uint8)
        polygon = np.array([[145, 175], [870, 120], [895, 595], [120, 650]], dtype=np.int32)
        cv2.fillConvexPoly(image, polygon, (242, 242, 238))
        cv2.polylines(image, [polygon], True, (25, 25, 25), 5)

        detections = OpenCvDocumentProposer().detect(image)

        self.assertTrue(detections)
        self.assertEqual(detections[0].source, "opencv_proposal")
        self.assertLess(detections[0].confidence, 0.61)

    def test_blank_scene_has_no_candidate(self):
        image = np.full((800, 1000, 3), 110, dtype=np.uint8)
        self.assertEqual(OpenCvDocumentProposer().detect(image), [])
