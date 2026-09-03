from pathlib import Path
import unittest

from medflow_release.config import ConfigError, load_config, resolve_path


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]

    def test_paper_configs_have_explicit_tmg_policy(self):
        expected = {
            "mimic_mortality.yaml": "minority_only",
            "mimic_icu_los.yaml": "minority_only",
            "eicu_mortality.yaml": "minority_only",
            "eicu_icu_los.yaml": "minority_only",
            "apava.yaml": "per_class",
            "ptb.yaml": "per_class",
        }
        for name, policy in expected.items():
            config = load_config(self.root / "configs" / "paper" / name)
            self.assertEqual(config["sampling"]["tmg"]["policy"], policy)
            self.assertTrue(resolve_path(config, config["paths"]["data_dir"]).is_absolute())

    def test_rejects_unknown_tmg_policy(self):
        with self.assertRaises(ConfigError):
            load_config(self.root / "tests" / "fixtures" / "invalid_tmg_policy.yaml")


if __name__ == "__main__":
    unittest.main()
