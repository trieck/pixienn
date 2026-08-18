import json
import os
import tempfile
import unittest
from pathlib import Path

from pixienn import Model


class BatchMosaicTests(unittest.TestCase):
    @staticmethod
    def _model(batch=2):
        # A detector is not needed to exercise the native batch/mosaic path.
        # Keeping the graph small makes this an inexpensive CPU integration test.
        return Model(3, 8, 8, batch=batch).maxpool(2, 2).build()

    def test_batch_list_writes_mosaic_and_returns_json(self):
        root = Path(__file__).parents[2]
        image = root / "resources/images/dog.jpg"

        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            image_list = work / "images.txt"
            # Include whitespace and a blank line to cover list normalization.
            image_list.write_text(f"  {image}\n\n{image}\n{image}\n")

            previous = os.getcwd()
            try:
                os.chdir(work)
                result = self._model().predict_batch_image_list(
                    image_list, confidence=0.1, nms_threshold=0.3)
            finally:
                os.chdir(previous)

            document = json.loads(result)
            self.assertEqual(document["type"], "FeatureCollection")
            self.assertEqual(document["features"], [])

            mosaic = work / "predictions.jpg"
            self.assertTrue(mosaic.is_file())
            # Three 640x480 tiles are laid out in a 2x2 grid.
            import cv2
            rendered = cv2.imread(str(mosaic))
            self.assertIsNotNone(rendered)
            self.assertEqual((rendered.shape[1], rendered.shape[0]), (1280, 960))

    def test_relative_paths_are_resolved_from_list_directory(self):
        root = Path(__file__).parents[2]
        source = root / "resources/images/dog.jpg"

        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            (work / "images").mkdir()
            image = work / "images/dog.jpg"
            image.write_bytes(source.read_bytes())
            image_list = work / "nested.txt"
            image_list.write_text("images/dog.jpg\n")

            previous = os.getcwd()
            try:
                os.chdir(work)
                result = self._model(batch=1).predict_batch_image_list(image_list)
            finally:
                os.chdir(previous)

            self.assertEqual(json.loads(result)["features"], [])
            self.assertTrue((work / "predictions.jpg").is_file())

    def test_unbuilt_model_is_rejected(self):
        with self.assertRaises(ValueError):
            Model(3, 8, 8).predict_batch_image_list("images.txt")

    def test_missing_and_empty_lists_are_rejected(self):
        model = self._model()
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            with self.assertRaises(Exception):
                model.predict_batch_image_list(work / "missing.txt")
            empty = work / "empty.txt"
            empty.write_text("\n  \n")
            with self.assertRaises(Exception):
                model.predict_batch_image_list(empty)


if __name__ == "__main__":
    unittest.main()
