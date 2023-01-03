#!/usr/bin/env bash

# install python3.10
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update -y
sudo apt upgrade -y
sudo apt install python3.10 python3.10-distutils -y

# install pip
curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# install poetry
curl -sSL https://install.python-poetry.org | python3.10 -
export PATH="~/.local/bin:$PATH"

# clone & install all requirements
git clone https://github.com/quasar-kim/kc-moe
cd kc-moe
poetry install --no-root --all-extra

# clone flaxformer
git clone https://github.com/google/flaxformer
mv flaxformer flaxformer-tmp
mv flaxformer-tmp/flaxformer .
rm -rf flaxformer-tmp
