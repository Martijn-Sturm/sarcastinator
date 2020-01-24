#!/bin/bash

#$ -M m.b.j.vandenberg@students.uu.nl
#$ -m bes
#$ -N infombd_project_cascade_validation
#$ -q all.q

# Merge stderr and stdout
#$ -j yes

# Limit runtime to 18 hours
#$ -l h_rt=18:00:00
#$ -l s_rt=18:00:00

set -euxo pipefail

cd /scratch/5636450/sarcastinator3
source venv/bin/activate

cd src

ls mainbalancedpickle.p
ls input_data/embs/{topic,user,word}_embs.p
ls input_data/train/{author_train,topic_train,x,y}.p
ls input_data/test/{author_test,topic_test,word_embs,x,y}.p
ls tensor/{test,train}/{topic,user,x}_tensor.npy

python tcn_test.py

# vim: et sw=2 sts=2
