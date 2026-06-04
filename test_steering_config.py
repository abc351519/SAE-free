import argparse
import unittest

from steering_config import format_steering_specs, parse_steering_specs


class SteeringConfigTest(unittest.TestCase):
    def test_parses_multiple_steer_specs(self):
        args = argparse.Namespace(
            steer=["9:1,2", "20:3,4"],
            layer_idx=31,
            feature_idx=[5],
        )

        self.assertEqual(parse_steering_specs(args), {9: [1, 2], 20: [3, 4]})

    def test_merges_repeated_layers_preserving_order(self):
        args = argparse.Namespace(
            steer=["9:1,2", "9:3"],
            layer_idx=31,
            feature_idx=[5],
        )

        self.assertEqual(parse_steering_specs(args), {9: [1, 2, 3]})

    def test_falls_back_to_existing_single_layer_arguments(self):
        args = argparse.Namespace(
            steer=None,
            layer_idx=31,
            feature_idx=[1810, 13375],
        )

        self.assertEqual(parse_steering_specs(args), {31: [1810, 13375]})

    def test_rejects_bad_specs(self):
        bad_specs = ["9", "9:", "x:1", "9:a", "-1:1", "9:-1"]
        for spec in bad_specs:
            with self.subTest(spec=spec):
                args = argparse.Namespace(steer=[spec], layer_idx=31, feature_idx=[1810])
                with self.assertRaises(ValueError):
                    parse_steering_specs(args)

    def test_formats_specs_for_safe_filenames(self):
        specs = {9: [1, 2], 20: [3], 31: [1810, 13375]}

        self.assertEqual(
            format_steering_specs(specs),
            "L9-1_2__L20-3__L31-1810_13375",
        )


if __name__ == "__main__":
    unittest.main()
