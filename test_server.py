import os
import requests


url = "http://127.0.0.1:8000/"  # change with actual banana model URL
path_img = "./resources/n01667114_mud_turtle.jpeg"

with open(path_img, "rb") as img:
    name_img = os.path.basename(path_img)
    files = {"image": (name_img, img, "multipart/form-data", {"Expires": "0"})}
    with requests.Session() as s:
        r = s.post(url, files=files)
        print(r.json())
