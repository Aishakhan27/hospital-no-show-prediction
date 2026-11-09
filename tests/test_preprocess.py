
import unittest
import pandas as pd
from pathlib import Path

# Locate the processed dataset files.
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"


class TestPreprocessing(unittest.TestCase):

    # Load the three processed datasets before running tests.
    @classmethod
    def setUpClass(cls):
        cls.train = pd.read_csv(DATA_DIR / "train.csv")
        cls.validation = pd.read_csv(DATA_DIR / "validation.csv")
        cls.test = pd.read_csv(DATA_DIR / "test.csv")

    # Check whether all three processed files exist.
    def test_files_exist(self):
        self.assertTrue((DATA_DIR / "train.csv").exists())
        self.assertTrue((DATA_DIR / "validation.csv").exists())
        self.assertTrue((DATA_DIR / "test.csv").exists())

    # Check that appointment IDs are not duplicated.
    def test_no_duplicate_appointments(self):
        for dataset in [self.train, self.validation, self.test]:
            self.assertEqual(dataset["appointment_id"].duplicated().sum(), 0)

    # Check that the datasets follow chronological order.
    def test_chronological_split(self):
        train = pd.to_datetime(self.train["booking_datetime"])
        validation = pd.to_datetime(self.validation["booking_datetime"])
        test = pd.to_datetime(self.test["booking_datetime"])

        self.assertLessEqual(train.max(), validation.min())
        self.assertLessEqual(validation.max(), test.min())

    # Check that the target contains only valid values.
    def test_valid_target(self):
        for dataset in [self.train, self.validation, self.test]:
            self.assertTrue(dataset["no_show"].isin([0, 1]).all())


if __name__ == "__main__":
    unittest.main()
