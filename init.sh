#!/bin/bash
sudo apt-get update
sudo apt-get upgrade -y
sudo apt-get install python3-venv -y
python3 -m venv env
source env/bin/activate

pip3 --no-cache-dir install https://github.com/mind/wheels/releases/download/tf1.4.1-cpu/tensorflow-1.4.1-cp36-cp36m-linux_x86_64.whl
pip3 install "gym==0.12.1"
pip3 uninstall numpy -y
pip3 install "numpy<1.17.0"
pip3 install "chainer<7.0.0"