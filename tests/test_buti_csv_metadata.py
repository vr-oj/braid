import csv
import tempfile
import unittest
from pathlib import Path

from processing.data_loader import extract_buti_settings, parse_and_validate_csv


class ButiCsvMetadataTests(unittest.TestCase):
    def test_reads_burst_optional_settings_without_changing_telemetry(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "run_force.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(
                    [
                        "time_s",
                        "frame_index",
                        "distance",
                        "cycle",
                        "force",
                        "experiment_type",
                        "preload_mm",
                        "deformation_mm",
                        "deformation_percent",
                        "steps",
                        "rate_forward_mm_s",
                        "rate_reverse_mm_s",
                        "cycles_configured",
                        "wire_diameter_mm",
                        "constant_tension_mn",
                    ]
                )
                writer.writerow(
                    [
                        0.25,
                        1,
                        0.1,
                        1,
                        2.5,
                        "Constant Velocity",
                        4.7,
                        0.94,
                        20,
                        1880,
                        0.1,
                        0.25,
                        10,
                        0.325,
                        125.0,
                    ]
                )

            data = parse_and_validate_csv(csv_path)

        self.assertEqual(data["time_s"], [0.25])
        self.assertEqual(data["force"], [2.5])
        self.assertEqual(
            data["buti_settings"],
            {
                "experiment_type": "Constant Velocity",
                "preload_mm": 4.7,
                "deformation_mm": 0.94,
                "deformation_percent": 20.0,
                "steps": 1880,
                "rate_forward_mm_s": 0.1,
                "rate_reverse_mm_s": 0.25,
                "cycles": 10,
                "wire_diameter_mm": 0.325,
                "constant_tension_mn": 125.0,
            },
        )

    def test_accepts_legacy_five_column_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "legacy_force.csv"
            csv_path.write_text(
                "time_s,frame_index,distance,cycle,force\n0.0,0,1.0,0,2.0\n",
                encoding="utf-8",
            )
            data = parse_and_validate_csv(csv_path)

        self.assertEqual(data["frame_index"], [0.0])
        self.assertNotIn("buti_settings", data)

    def test_normalizes_repeated_tiff_settings(self):
        settings = extract_buti_settings(
            {
                "buti_settings": [
                    {"preload_mm": 4.7, "cycles": 10},
                    {"preload_mm": 4.7, "cycles": 10},
                ]
            }
        )
        self.assertEqual(settings, {"preload_mm": 4.7, "cycles": 10})


if __name__ == "__main__":
    unittest.main()
