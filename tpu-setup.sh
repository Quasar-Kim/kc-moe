# install python3.10
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update -y
sudo apt upgrade -y
sudo apt install python3.10 python3.10-distutils -y

# install pip
curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# install t5x
# (will also install JAX[tpu])
git clone https://github.com/google-research/t5x
python3.10 -m pip install -e ./t5x[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# clone repo
git clone https://github.com/quasar-kim/kc-moe
cd kc-moe

# install flaxformer
git clone https://github.com/google/flaxformer
python3.10 -m pip install ./flaxformer
mv flaxformer flaxformer-tmp
mv flaxformer-tmp/flaxformer .
rm -rf flaxformer-tmp

python3.10 -m pip install python-mecab-ko
