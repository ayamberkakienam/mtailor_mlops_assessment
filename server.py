# Taken from: https://github.com/lucataco/serverless-template-anything-v4.0/blob/main/server.py
# Do not edit if deploying to Banana Serverless
# This file is boilerplate for the http server, and follows a strict interface.

# Instead, edit the init() and inference() functions in app.py

import io
import subprocess
from PIL import Image
from sanic import Sanic, response

from model import preprocess_numpy, ONNXModel

# We do the model load-to-GPU step on server startup
# so the model object is available globally for reuse
onnx_model = ONNXModel("./onnx_model.onnx")

# Create the http server app
server = Sanic("my_app")

# Healthchecks verify that the environment is correct on Banana Serverless
@server.route("/healthcheck", methods=["GET"])
def healthcheck(request):
    # dependency free way to check if GPU is visible
    gpu = False
    out = subprocess.run("nvidia-smi", shell=True)
    if out.returncode == 0:  # success state on shell command
        gpu = True

    return response.json({"state": "healthy", "gpu": gpu})


# Inference POST handler at '/' is called for every http call from Banana
@server.route("/", methods=["POST"])
def inference(request):
    file = request.files["image"][0].body
    img = Image.open(io.BytesIO(file))
    img = preprocess_numpy(img)
    prediction = onnx_model.predict(img)

    return response.json({"class_id": int(prediction)})
    #try:
    #    model_inputs = response.json.loads(request.json)
    #except:
    #    model_inputs = request.json

    #output = user_src.inference(model_inputs)

    #return response.json(output)


if __name__ == "__main__":
    server.run(host="0.0.0.0", port=8000, workers=1)
