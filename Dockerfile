# Taken from: https://github.com/lucataco/serverless-template-anything-v4.0/blob/main/server.py
# Must use a Cuda version 11+
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

WORKDIR /

# Install git
RUN apt-get update && apt-get install -y git
RUN conda install pytorch torchvision pytorch-cuda=11.6 -c pytorch -c nvidia

# Install python packages
RUN pip3 install --upgrade pip
ADD requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

# Include the codebase into the image
COPY . .

# Expose docker port
EXPOSE 8000

CMD python3 -u server.py
