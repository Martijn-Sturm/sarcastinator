#!/bin/bash

#$ -M m.b.j.vandenberg@students.uu.nl
#$ -m bes
#$ -N infombd_project_cascade_tune3
#$ -q long.q

# Merge stderr and stdout
#$ -j yes

# Limit runtime to 18 hours
#$ -l h_rt=18:00:00
#$ -l s_rt=18:00:00

set -euxo pipefail

# Create working directory and enter it
mkdir "/scratch/5636450-infombd"

function cleanup {
  # Collect results, save to home directory
  zip -j ~/infombd-tune3-results.zip /scratch/5636450-infombd/sarcastinator/src/{logs,results,model}/* || echo "Zipping failed!"

  # Exit directory
  cd ~

  # Remove scratch space
  rm -rf /scratch/5636450-infombd
}
trap cleanup EXIT

cd "/scratch/5636450-infombd"

# Download fasttext embeddings and extract
curl --silent --show-error -O https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip
unzip crawl-300d-2M.vec.zip -d "/scratch/5636450-infombd"

curl --silent --show-error -o user_gcca_embeddings.npz https://files.maartenberg.nl/selif/usergccaembeddings.npz

curl --silent --show-error -O https://nlp.cs.princeton.edu/SARC/2.0/main/comments.json.bz2
bunzip2 comments.json.bz2

sha256sum --check --strict <<EOSUMS
c85534cb2b7f816cb8d564129add1454a7a789d7b607af7aaa4e0d1c927e62ed  crawl-300d-2M.vec
5bfffffbabdab299d4c9165c47275e8f982807a6eaca37ee1f71d3a79ddb544d  crawl-300d-2M.vec.zip
c8b57a3e1662fb2494bcf9010d16c786cf214586210e67f98923b0d0a0997f92  user_gcca_embeddings.npz
15a7b2faba2f42695a27de7db8f3a0840d7997da74caa3245d9669774787a4a4  comments.json
EOSUMS

eval $(ssh-agent)
ssh-add ~/.ssh/sarcastinator-deploy-key
git clone git@github.com:Martijn-Sturm/sarcastinator.git sarcastinator
ln -s /scratch/5636450-infombd/user_gcca_embeddings.npz /scratch/5636450-infombd/sarcastinator/users/user_embeddings/user_gcca_embeddings.npz
ln -s /scratch/5636450-infombd/comments.json /scratch/5636450-infombd/sarcastinator/data/comments.json

cd sarcastinator

python3.6 -m venv venv
source venv/bin/activate
pip install -U pip setuptools wheel

pip install -r requirements2.txt

cd src
python process_data.py "/scratch/5636450-infombd/crawl-300d-2M.vec"

ls mainbalancedpickle.p

python prepare.py

ls input_data/embs/{topic,user,word}_embs.p
ls input_data/train/{author_train,topic_train,x,y}.p
ls input_data/test/{author_test,topic_test,word_embs,x,y}.p

ls tensor/{test,train}/{topic,user,x}_tensor.npy

python tcn_test.py

# vim: et sw=2 sts=2
