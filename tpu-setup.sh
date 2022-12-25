# run this script in tpu vm shell
# right after creating the vm

# install python3.10
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update -y
sudo apt upgrade -y
sudo apt install python3.10 -y

# install poetry
curl -sSL https://install.python-poetry.org | python3.10 -

# install requirements
# --all-extras will install packages only required for running experiments
poetry install --no-root --all-extras