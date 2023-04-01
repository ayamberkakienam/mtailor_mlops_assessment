import unittest
from PIL import Image

from model import ONNXModel, preprocess_numpy


class TestModel(unittest.TestCase):
    def test_predict(self):
        img_turtle = preprocess_numpy(
            Image.open("./resources/n01667114_mud_turtle.jpeg")
        )
        onnx_model = ONNXModel("./onnx_model.onnx")
        predict_turtle = onnx_model.predict(img_turtle)
        self.assertEqual(predict_turtle, 35)

        img_fish = preprocess_numpy(Image.open("./resources/n01440764_tench.jpeg"))
        onnx_model = ONNXModel("./onnx_model.onnx")
        predict_fish = onnx_model.predict(img_fish)
        self.assertEqual(predict_fish, 0)


if __name__ == "__main__":
    unittest.main()
