# MTailor MLOps Assessment

This repository provide the means to converts a PyTorch model into ONNX model and deploy
it on banana.dev as a Docker container.

# Pre-requisites
1. Install the required library in `requirements.txt`.
2. Download the model weight from this [link](https://www.dropbox.com/s/b7641ryzmkceoc9/pytorch_model_weights.pth?dl=0).

# Feature(s)
The following are the functionalities currently implemented in this repo:

## 1. Convert PyTorch model to ONNX model
This feature import the previously implemented PyTorch model and export it in ONNX 
format.

To use it, make sure the file `pytorch_model_weights.pth` is already downloaded on the 
project root directory, then call the script:
```
python convert_to_onnx.py
```
It will create an file named `onnx_model.onnx`.

## 2. Run image classification using ONNX model
To use ONNX model for prediction, make sure you already export the ONNX model first.
```
from PIL import Image

from model import ONNXModel, preprocess_numpy

img = preprocess_numpy(Image.open("./resources/n01667114_mud_turtle.jpeg"))
onnx_model = ONNXModel("./onnx_model.onnx")
prediction = onnx_model.predict(img)
print(prediction)
```

## 3. Run model as a server
To run model as a server, make sure you already export the ONNX model first,
then run the following script
```
python server.py
```

The server will run in URL http://0.0.0.0:8000. You can test the server by using the 
script `test_server.py`. Make sure to put the actual URL in the `url` variable inside
the script first.
```
python test_server.py

Response:
{'class_id': 35}
```
