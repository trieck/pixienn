import unittest
import json
import tempfile
import os
import subprocess
from array import array
from pathlib import Path

from pixienn import Model


class ModelTests(unittest.TestCase):
    @staticmethod
    def _yolov1_tiny(device="cpu"):
        from pixienn import Model

        root = Path(__file__).parents[2]
        labels = (root / "resources/data/voc.names").read_text().splitlines()
        model = Model(3, 448, 448, device=device).set_labels(labels)
        for filters in (16, 32, 64, 128, 256, 512):
            model.conv(filters, 3, pad=True, stride=1, activation="leaky",
                       batch_normalize=True).maxpool(2, 2)
        return (model
                .conv(1024, 3, pad=True, stride=1, activation="leaky", batch_normalize=True)
                .conv(256, 3, pad=True, stride=1, activation="leaky", batch_normalize=True)
                .connected(1470, activation="linear")
                .detection(log_interval=50, class_scale=1, coord_scale=5, coords=4,
                           noobject_scale=0.5, num=2, object_scale=1, rescore=True,
                           side=7, softmax=False, sqrt=True))

    def test_builds_layers_from_python_definitions(self):
        model = Model(channels=3, height=8, width=8)
        model.add_layer({"type": "maxpool", "kernel": 2, "stride": 2})
        self.assertFalse(model.built)

        model.build()

        self.assertTrue(model.built)
        self.assertEqual(model.layer_count, 1)
        self.assertEqual(model.input_shape, [3, 8, 8])
        self.assertEqual(model.output_shape, [3, 4, 4])

    def test_rejects_changes_after_build(self):
        model = Model(3, 8, 8)
        model.build()
        with self.assertRaises(ValueError):
            model.add_layer({"type": "maxpool", "kernel": 2, "stride": 2})

    def test_python_first_graph_spec_and_helpers(self):
        model = (Model.from_spec({"channels": 3, "height": 8, "width": 8})
                 .conv(4, 3, pad=True, activation="leaky")
                 .maxpool(2, 2))
        self.assertEqual(model.spec["layers"][0]["type"], "conv")
        self.assertEqual(model.spec["layers"][1]["type"], "maxpool")
        model.build()
        self.assertEqual(model.output_shape, [4, 4, 4])

    def test_forward_returns_tensor_with_inferred_shape(self):
        model = Model(1, 4, 4).maxpool(2, 2).build()
        output = model.forward(__import__("pixienn").Tensor([1, 1, 4, 4], 1.0))
        self.assertEqual(output.shape, [1, 1, 2, 2])
        self.assertEqual(output.values(), [1.0] * 4)

    def test_forward_accepts_float_buffer(self):
        model = Model(1, 4, 4).maxpool(2, 2).build()
        output = model.forward(array("f", [1.0] * 16))
        self.assertEqual(output.values(), [1.0] * 4)

    def test_connected_and_softmax_helpers(self):
        model = Model(1, 2, 2).connected(3).softmax(detector=False).build()
        self.assertEqual(model.layer_count, 2)
        self.assertEqual(model.layer_shapes, [[3, 1, 1], [3, 1, 1]])

    def test_detector_helpers_are_python_composable(self):
        model = Model(3, 13, 13).yolo([10, 13, 16, 30, 33, 23], [0, 1, 2], classes=2)
        self.assertEqual(model.layers[0]["type"], "yolo")
        self.assertEqual(model.layers[0]["mask"], [0, 1, 2])

    def test_complete_yolov1_tiny_matches_cpp_yaml_inference(self):
        root = Path(__file__).parents[2]
        binary = root / "cmake-build-release-cuda/bin/pixienn"
        self.assertTrue(binary.exists(), f"missing native inference binary: {binary}")
        image = root / "resources/images/dog.jpg"
        config = root / "resources/cfg/yolov1-tiny-cfg.yml"

        model = self._yolov1_tiny().build()
        self.assertEqual(model.layer_count, 16)
        self.assertEqual([layer["dilation"] for layer in model.layers if layer["type"] == "conv"], [1] * 8)
        self.assertEqual(model.layer_shapes[-2:], [[1470, 1, 1], [1470, 1, 1]])
        model.load_weights(root / "resources/weights/yolov1-tiny.weights").set_threshold(0.2)

        with tempfile.TemporaryDirectory() as cpp_dir, tempfile.TemporaryDirectory() as python_dir:
            subprocess.run([
                str(binary), "--no-gpu", "--confidence=0.2", "--nms=0.3",
                str(config), str(image),
            ], cwd=cpp_dir, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

            previous = os.getcwd()
            try:
                os.chdir(python_dir)
                python_json = model.predict_image(image, nms_threshold=0.3,
                                                  geojson_path="predictions.geojson")
            finally:
                os.chdir(previous)

            cpp_json = json.loads(Path(cpp_dir, "predictions.geojson").read_text())
            self.assertEqual(json.loads(python_json), cpp_json)
            self.assertEqual(Path(python_dir, "predictions.geojson").read_bytes(),
                             Path(cpp_dir, "predictions.geojson").read_bytes())
            self.assertEqual(Path(python_dir, "predictions.jpg").read_bytes(),
                             Path(cpp_dir, "predictions.jpg").read_bytes())

    def test_evaluation_and_prediction_require_a_built_model(self):
        model = Model(1, 4, 4)
        with self.assertRaises(ValueError):
            model.evaluate()
        with self.assertRaises(ValueError):
            model.predict_json("missing.png")

    def test_weights_round_trip(self):
        model = Model(1, 4, 4).maxpool(2, 2).build()
        with tempfile.NamedTemporaryFile(suffix=".weights") as weights:
            model.save_weights(weights.name)
            restored = Model(1, 4, 4).maxpool(2, 2).build()
            restored.load_weights(weights.name)
            output = restored.forward(__import__("pixienn").Tensor([1, 1, 4, 4], 2.0))
            self.assertEqual(output.values(), [2.0] * 4)

    def test_mode_and_training_state_controls(self):
        model = Model(1, 4, 4)
        with self.assertRaises(ValueError):
            model.set_mode("unknown")
        model.maxpool(2, 2).build().set_mode("training")
        with tempfile.NamedTemporaryFile(suffix=".training") as state:
            model.save_training_state(state.name)
            self.assertTrue(__import__("pathlib").Path(state.name + ".training").exists())

    def test_runtime_properties_and_threshold(self):
        model = Model(3, 8, 8).maxpool(2, 2).build()
        self.assertEqual(model.channels, 3)
        self.assertEqual(model.height, 8)
        self.assertEqual(model.width, 8)
        self.assertEqual(model.layers[0]["type"], "maxpool")
        self.assertEqual(model.layer_shapes, [[3, 4, 4]])
        model.set_labels(["object"])
        self.assertEqual(model.labels, ["object"])
        self.assertEqual(model.mode, "inference")
        model.set_threshold(0.65)
        self.assertAlmostEqual(model.threshold, 0.65, places=5)

    def test_python_training_configuration_builds_native_training_graph(self):
        model = Model(
            1, 4, 4, batch=2, subdivisions=1, max_batches=10,
            learning_rate={"initial_learning_rate": 0.001, "policy": "constant"},
        )
        model.configure_training(**{
            "train-images": "/tmp/train.txt",
            "train-labels": "/tmp/labels",
            "val-images": "/tmp/val.txt",
            "val-labels": "/tmp/val-labels",
        }).maxpool(2, 2).build()
        self.assertEqual(model.batch, 2)
        self.assertEqual(model.output_shape, [1, 2, 2])
        self.assertEqual(model.training_config["train-images"], "/tmp/train.txt")

        adam = Model(
            1, 4, 4, max_batches=10,
            learning_rate={"initial_learning_rate": 0.001, "policy": "constant"},
            adam={"enabled": True, "beta1": 0.8, "beta2": 0.95, "epsilon": 1e-7},
        ).configure_training(**{
            "train-images": "/tmp/train.txt", "train-labels": "/tmp/labels",
            "val-images": "/tmp/val.txt", "val-labels": "/tmp/val-labels",
        }).maxpool(2, 2).build()
        self.assertTrue(adam.adam_enabled)
        self.assertAlmostEqual(adam.adam_beta1, 0.8)

    def test_complete_config_round_trip_is_json_yaml_compatible(self):
        model = Model(3, 8, 8).maxpool(2, 2)
        config = model.config(configuration={"model": "model.json", "labels": "labels.txt"},
                              training={"train-images": "train.txt"})
        self.assertEqual(config["model"]["layers"][0]["type"], "maxpool")
        with tempfile.NamedTemporaryFile(suffix=".yaml") as output:
            model.save_config(output.name, configuration=config["configuration"],
                              training=config["training"])
            self.assertEqual(json.load(output), config)

    def test_complete_config_can_rebuild_graph(self):
        model = Model(1, 4, 4).maxpool(2, 2)
        with tempfile.NamedTemporaryFile(suffix=".json") as output:
            model.save_config(output.name, configuration={"model": "unused", "labels": "unused"})
            rebuilt = Model.from_config(output.name).build()
            self.assertEqual(rebuilt.output_shape, [1, 2, 2])

    def test_native_two_file_export_rewrites_model_reference(self):
        model = Model(1, 4, 4).maxpool(2, 2)
        with tempfile.TemporaryDirectory() as directory:
            config_path = model.save_native_files(
                directory, configuration={"model": "placeholder", "labels": "labels.txt"})
            with open(config_path) as stream:
                document = json.load(stream)
            self.assertEqual(document["configuration"]["model"], "model.json")
            with open(config_path.parent / "model.json") as stream:
                model_document = json.load(stream)
            self.assertEqual(model_document["model"]["layers"][0]["type"], "maxpool")


if __name__ == "__main__":
    unittest.main()
