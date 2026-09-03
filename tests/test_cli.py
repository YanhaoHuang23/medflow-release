from pathlib import Path
import unittest

from medflow_release.cli import _generator_dataset_args, _implementation_root, _selector_args, _tmg_args
from medflow_release.config import load_config


class CliMappingTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]

    def test_generator_has_its_own_tuple_arguments(self):
        config = load_config(self.root / "configs" / "paper" / "ptb.yaml")
        args = _generator_dataset_args(config)
        self.assertIn("--data-dir", args)
        self.assertNotIn("--dataname", args)
        self.assertEqual(args[args.index("--input-dim") + 1], "15")

    def test_tmg_policy_maps_to_legacy_label_mode(self):
        ehr = load_config(self.root / "configs" / "paper" / "mimic_mortality.yaml")
        signal = load_config(self.root / "configs" / "paper" / "apava.yaml")
        ehr_args = _tmg_args(ehr)
        signal_args = _tmg_args(signal)
        self.assertEqual(ehr_args[ehr_args.index("--token-manifold-guidance-labels") + 1], "target")
        self.assertEqual(signal_args[signal_args.index("--token-manifold-guidance-labels") + 1], "all")

    def test_bundled_implementation_root_is_the_release_directory(self):
        root = _implementation_root(None)
        self.assertEqual(root, self.root)
        self.assertTrue((root / "train_msvq.py").is_file())
        self.assertTrue((root / "scripts" / "generate_msdflow_mimic.py").is_file())

    def test_ehr_selector_is_explicitly_configured(self):
        config = load_config(self.root / "configs" / "paper" / "mimic_mortality.yaml")
        args = _selector_args(config)
        self.assertIn("--posthoc-selector", args)
        self.assertIn("influence_utility", args)


if __name__ == "__main__":
    unittest.main()
