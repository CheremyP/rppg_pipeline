import urllib.request as urlreq
import os
import torch
import torchvision
import subprocess
import sys

#TODO: Download the Haar Cascade file


LBFmodel_url = "https://github.com/kurnianggoro/GSOC2017/raw/master/data/lbfmodel.yaml"

# save facial landmark detection model's name as LBFmodel
LBFmodel = "lbfmodel.yaml"

# check if file is in working directory
if not (LBFmodel in os.listdir(os.curdir)):
    # download picture from url and save locally as lbfmodel.yaml, < 54MB
    urlreq.urlretrieve(LBFmodel_url, LBFmodel)
    print("File downloaded")

print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA is available:", torch.cuda.is_available())

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install("opencv-python")
install("matplotlib")
install("onnx")
install("onnxruntime-gpu")
install("av")
install("git+https://github.com/facebookresearch/segment-anything.git")



subprocess.run(["wget", "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"])