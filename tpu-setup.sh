#!/usr/bin/env bash

# install python3.10
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update -y
sudo apt upgrade -y
sudo apt install python3.10 python3.10-distutils python3.10-venv -y

# install pip
curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# install poetry
curl -sSL https://install.python-poetry.org | python3.10 -
export PATH="~/.local/bin:$PATH"

# clone & install all requirements
git clone https://github.com/quasar-kim/kc-moe -b europe --single-branch --recurse-submodules
cd kc-moe
rm poetry.lock
poetry install --no-root --all-extras