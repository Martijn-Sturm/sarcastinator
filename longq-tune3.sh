#!/bin/bash

#$ -M m.b.j.vandenberg@students.uu.nl
#$ -m bes
#$ -N infombd_project_cascade_tune3
#$ -q long.q

set -euxo pipefail

# Create working directory and enter it
mkdir "/scratch/5636450-infombd"

function cleanup {
  # Collect results, save to home directory
  zip -j ~/infombd-tune3-results.zip /scratch/5636450-infombd/sarcastinator/src/{logs,results}/* || echo "Zipping failed!"

  # Exit directory
  cd ~

  # Remove scratch space
  rm -rf /scratch/5636450-infombd
}
trap cleanup EXIT

cd "/scratch/5636450-infombd"

# Download fasttext embeddings and extract
wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip
unzip crawl-300d-2M.vec.zip -d $SCRATCH

sha256sum --check --strict <<EOSUMS
c85534cb2b7f816cb8d564129add1454a7a789d7b607af7aaa4e0d1c927e62ed  crawl-300d-2M.vec
5bfffffbabdab299d4c9165c47275e8f982807a6eaca37ee1f71d3a79ddb544d  crawl-300d-2M.vec.zip
EOSUMS

ssh-add ~/.ssh/sarcastinator-deploy-key
git clone git@github.com:Martijn-Sturm/sarcastinator.git sarcastinator

cd sarcastinator

python3.6 -m venv venv
source venv/bin/activate
pip install -U pip setuptools wheel

pip install -r requirements2.txt

cd src
python process_data.py "${SCRATCH}/crawl-300d-2M.vec"

ls mainbalancedpickle.p

python prepare.py

ls input_data/embs/{topic,user,word}_embs.p
ls input_data/train/{author_train,topic_train,x,y}.p
ls input_data/test/{author_test,topic_test,word_embs,x,y}.p

ls src/tensor/{test,train}/{topic,user,x}_tensor.npy

python tcn_tune3.py

# vim: et sw=2 sts=2
