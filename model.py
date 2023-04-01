import numpy as np
import onnxruntime
from PIL import Image
from torch import Tensor
from torchvision import transforms


def preprocess_numpy(img: Image) -> Tensor:
    # import preprocessing function from pytorch_model.Classifier class
    resize = transforms.Resize((224, 224))
    crop = transforms.CenterCrop((224, 224))
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    img = resize(img)
    img = crop(img)
    img = to_tensor(img)
    img = normalize(img)
    return img.unsqueeze(0)


class ONNXModel:
    def __init__(self, onnx_file: str):
        self._session = onnxruntime.InferenceSession(onnx_file)

    def _to_numpy(self, tensor: Tensor) -> np.ndarray:
        return (
            tensor.detach().cpu().numpy()
            if tensor.requires_grad
            else tensor.cpu().numpy()
        )

    def predict(self, image: Image) -> int:
        inputs = {self._session.get_inputs()[0].name: self._to_numpy(image)}
        outputs = self._session.run(None, inputs)
        # get the highest probability in `outputs` as the predicted classification
        final_out = np.array(outputs).argmax()
        return final_out


if __name__ == "__main__":
    # Load and prepare image
    img = Image.open("./resources/n01667114_mud_turtle.jpeg")
    img = preprocess_numpy(img)

    # Load and run the model
    onnx_model = ONNXModel("./onnx_model.onnx")
    prediction = onnx_model.predict(img)
    print(prediction)
