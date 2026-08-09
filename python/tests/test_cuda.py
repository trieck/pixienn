import unittest
import json
import os
import tempfile
from pathlib import Path

from pixienn import CudaTensor, Model, Tensor
from test_model import ModelTests


CUDA_AVAILABLE = CudaTensor is not None


@unittest.skipUnless(CUDA_AVAILABLE, "PixieNN Python bindings were built without CUDA")
class CudaTests(unittest.TestCase):
    def test_cuda_tensor_and_model_match_cpu(self):
        cpu_model = Model(1, 4, 4).maxpool(2, 2).build()
        cuda_model = Model(1, 4, 4, device="cuda").maxpool(2, 2).build()

        cpu_input = Tensor([1, 1, 4, 4], 1.0)
        cuda_input = CudaTensor([1, 1, 4, 4], 1.0)

        self.assertEqual(cuda_input.device, "cuda")
        self.assertEqual(cuda_model.device, "cuda")
        self.assertEqual(cuda_model.forward(cuda_input).device, "cuda")
        self.assertEqual(cuda_model.forward(cuda_input).values(),
                         cpu_model.forward(cpu_input).values())

    def test_yolov1_tiny_cuda_matches_cpu_on_canonical_image(self):
        root = Path(__file__).parents[2]
        weights = root / "resources/weights/yolov1-tiny.weights"
        image = root / "resources/images/dog.jpg"
        cpu_model = ModelTests._yolov1_tiny("cpu").build().load_weights(weights).set_threshold(0.2)
        cuda_model = ModelTests._yolov1_tiny("cuda").build().load_weights(weights).set_threshold(0.2)
        with tempfile.TemporaryDirectory() as cpu_dir, tempfile.TemporaryDirectory() as cuda_dir:
            previous = os.getcwd()
            try:
                os.chdir(cpu_dir)
                cpu_json = cpu_model.predict_image(image, nms_threshold=0.3, geojson_path="predictions.geojson")
                os.chdir(cuda_dir)
                cuda_json = cuda_model.predict_image(image, nms_threshold=0.3, geojson_path="predictions.geojson")
            finally:
                os.chdir(previous)
            cpu_result = json.loads(cpu_json)
            cuda_result = json.loads(cuda_json)
            self.assertEqual(len(cuda_result["features"]), len(cpu_result["features"]))
            for cpu_feature, cuda_feature in zip(cpu_result["features"], cuda_result["features"]):
                self.assertEqual(cuda_feature["properties"]["class"], cpu_feature["properties"]["class"])
                self.assertAlmostEqual(cuda_feature["properties"]["confidence"],
                                       cpu_feature["properties"]["confidence"], places=3)
                for cpu_point, cuda_point in zip(cpu_feature["geometry"]["coordinates"][0],
                                                 cuda_feature["geometry"]["coordinates"][0]):
                    self.assertAlmostEqual(cuda_point[0], cpu_point[0], delta=0.1)
                    self.assertAlmostEqual(cuda_point[1], cpu_point[1], delta=0.1)
            self.assertGreater(Path(cuda_dir, "predictions.jpg").stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
