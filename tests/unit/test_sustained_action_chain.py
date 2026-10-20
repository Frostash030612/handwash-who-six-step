import unittest

from handwash.errors import ProtocolError
from handwash.pipelines.action_chain import sustained_action_chain


class SustainedActionChainTests(unittest.TestCase):
    def chain(self, labels, smooth=0):
        return sustained_action_chain(labels, fps=5, min_action_seconds=3,
                                      smoothing_window_seconds=smooth, background_class=0)

    def test_short_island_is_rejected_without_extending_neighbours(self):
        result = self.chain([3]*30 + [4]*9 + [6]*30)
        self.assertEqual(result['order'], [3, 6])
        self.assertEqual([s['duration_s'] for s in result['segments']], [6, 1.8, 6])
        self.assertEqual(result['segments'][1]['status'], 'unconfirmed_short')

    def test_exact_threshold_and_irregular_order_are_preserved(self):
        self.assertEqual(self.chain([5]*15+[4]*15+[3]*15)['order'], [5, 4, 3])
        self.assertEqual(self.chain([5]*14)['order'], [])

    def test_empty_and_all_short_do_not_invent_a_step(self):
        self.assertEqual(self.chain([])['order'], [])
        self.assertEqual(self.chain([4]*9+[5]*9)['order'], [])

    def test_fragments_do_not_accumulate_to_cross_threshold(self):
        self.assertEqual(self.chain([4]*9+[0]+[4]*9)['order'], [])

    def test_one_frame_noise_does_not_fragment_long_action(self):
        raw = [4]*9+[5]+[4]*10
        self.assertEqual(self.chain(raw, smooth=1)['order'], [4])
        self.assertEqual(raw[9], 5)

    def test_real_long_repeated_step_is_retained(self):
        self.assertEqual(self.chain([4]*20+[3]*20+[4]*20)['order'], [4, 3, 4])

    def test_invalid_fps_is_rejected(self):
        with self.assertRaises(ProtocolError):
            sustained_action_chain([1], fps=0, min_action_seconds=3,
                                   smoothing_window_seconds=1, background_class=0)


if __name__ == '__main__':
    unittest.main()
