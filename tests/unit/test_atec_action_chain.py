import unittest

from handwash.errors import ProtocolError
from handwash.pipelines.action_chain import atec_action_chain


class ATECActionChainTests(unittest.TestCase):
    def chain(self, labels, confidences=None):
        return atec_action_chain(labels, fps=5, smoothing_window_seconds=0,
                                 background_class=0, winner_confidences=confidences)

    def test_relative_short_outlier_is_unconfirmed(self):
        labels = [1]*35+[2]*40+[3]*30+[4]*10+[5]*35+[6]*30
        confidence = [0.95]*len(labels)
        confidence[105:115] = [0.7]*10
        result = self.chain(labels, confidence)
        self.assertEqual(result['order'], [1, 2, 3, 5, 6])
        self.assertEqual(result['segments'][3]['status'], 'unconfirmed_outlier')

    def test_same_short_action_in_faster_video_is_confirmed(self):
        labels = [1]*9+[2]*10+[3]*8+[4]*10+[5]*9
        result = self.chain(labels, [0.9]*len(labels))
        self.assertEqual(result['order'], [1, 2, 3, 4, 5])

    def test_short_single_action_has_insufficient_comparison(self):
        result = self.chain([4]*5, [0.8]*5)
        self.assertEqual(result['order'], [4])
        self.assertFalse(result['duration_comparison_available'])

    def test_repeated_action_is_preserved_when_supported(self):
        labels = [4]*25+[3]*25+[4]*25+[5]*25
        self.assertEqual(self.chain(labels)['order'], [4, 3, 4, 5])

    def test_invalid_fps_is_rejected(self):
        with self.assertRaises(ProtocolError):
            atec_action_chain([1], fps=0, smoothing_window_seconds=1,
                              background_class=0)


if __name__ == '__main__':
    unittest.main()
