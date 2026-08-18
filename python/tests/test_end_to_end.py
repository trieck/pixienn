"""Small Python-first training/checkpoint/inference proof."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

# Prefer the repository's compiled extension and Python package over an older
# globally installed PixieNN package when this test is launched by an IDE.
REPO_ROOT = Path(__file__).resolve().parents[2]
for build in ("cmake-build-python", "cmake-build-debug-cuda", "cmake-build-release-cuda"):
    package_root = REPO_ROOT / build / "python"
    if list((package_root / "pixienn").glob("_native*.so")):
        sys.path.insert(0, str(package_root))
        break
sys.path.append(str(REPO_ROOT / "python"))

from pixienn import Model


class PythonEndToEndTests(unittest.TestCase):
    @staticmethod
    def _model(root: Path, *, training: bool, weights: Path | None = None,
               event_file: Path | None = None) -> Model:
        data = root / "resources/data/voc"
        labels = (root / "resources/data/voc.names").read_text().splitlines()

        options = {}
        if training:
            options.update({
                "max_batches": 1,
                "subdivisions": 1,
                "momentum": 0.9,
                "decay": 0.0005,
                "adam": {"enabled": True, "beta1": 0.9, "beta2": 0.999, "epsilon": 1e-8},
                "augmentation": {"enabled": False},
                "write_metrics_interval": 1,
                "learning_rate": {"initial_learning_rate": 0.001, "policy": "constant"},
            })
        if weights is not None:
            options.update({
                "weights-file": str(weights),
                "backup-dir": str(weights.parent / "backup"),
            })
        if event_file is not None:
            options["event_file"] = str(event_file)

        model = Model(3, 64, 64, batch=1, **options).set_labels(labels)
        if training:
            model.configure_training(**{
                "train-images": str(data / "train-1.txt"),
                "train-labels": str(data / "labels"),
                "val-images": str(data / "val-10.txt"),
                "val-labels": str(data / "labels"),
            })

        return (model
                .conv(16, 3, stride=2, pad=True, activation="leaky", batch_normalize=True)
                .conv(32, 3, stride=2, pad=True, activation="leaky", batch_normalize=True)
                .conv(32, 3, stride=1, pad=True, activation="leaky", batch_normalize=True)
                .conv(24, 1, pad=False, activation="linear")
                .centernet(max_detections=50)
                .build())

    def test_python_training_checkpoint_and_inference(self):
        root = REPO_ROOT
        image = root / "resources/data/voc/images/000508.jpg"

        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            weights = run / "model.weights"
            initial = run / "initial.weights"
            event_file = run / "events.tfevents"

            model = self._model(root, training=True, weights=weights, event_file=event_file)
            model.save_weights(initial)
            initial_bytes = initial.read_bytes()

            model.train()

            self.assertGreater(model.seen, 0)
            self.assertGreater(model.cost, 0.0)
            self.assertTrue(weights.exists())
            self.assertNotEqual(initial_bytes, weights.read_bytes())
            self.assertGreater(event_file.stat().st_size, 0)

            restored = self._model(root, training=False, weights=weights)
            restored.load_weights(weights).set_threshold(0.0)
            previous = os.getcwd()
            try:
                os.chdir(run)
                result = restored.predict_image(image, nms_threshold=0.3,
                                                geojson_path="predictions.geojson")
            finally:
                os.chdir(previous)

            document = json.loads(result)
            self.assertEqual(document["type"], "FeatureCollection")
            self.assertIsInstance(document["features"], list)
            self.assertTrue((run / "predictions.geojson").exists())
            self.assertGreater((run / "predictions.jpg").stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
