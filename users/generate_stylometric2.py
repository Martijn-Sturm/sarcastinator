#!/usr/bin/env python
import pandas as pd
import numpy as np
import csv
import gensim, os
import math

# Load data
# doc2vec = gensim.models.Doc2Vec.load('./models/user_stylometric.model')
data = np.asarray(pd.read_csv('./train_balanced_user.csv', header=None))

# Load universal sentence embedder function:
# From: https://tfhub.dev/google/universal-sentence-encoder/4
import tensorflow as tf 
import tensorflow_hub as hub
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# Preprocess sentences
# Delete '<END>' delimiters?

directory = "./user_embeddings"
if not os.path.exists(directory):
	os.makedirs(directory)

# Add report layer to see progress:
def modulo(n):
	return(int(n/100))

def report(index, mod):
	if index % mod == 0:
		print(str(int(index/mod))+"%")

def process(token, index, report, mod):
    if report:
        report(index, mod) 
    return token

# Generate stylometric user embeddings making use of universal sentence embedded function:
with open(directory+"/user_stylometric2.csv",'w') as file:
	wr = csv.writer(file, quoting=csv.QUOTE_ALL)

	# Inferring universal sentence embedding vectors for each user
	print("Starting inferring vectors for each user")
	mod_vec1 = modulo(int(data.shape[0]))
	vectors = np.asarray([process(embed([data[i][1]]), i, report, mod_vec1) for i in range(data.shape[0])])

	print("Staring matrix preparation")
	users = data[:,0]	
	for i in range(len(users)):
		ls=[]
		ls.append(users[i])
		v = [0]*100
		for j in range(len(vectors[i])):
			v[j] = vectors[i][j]
		ls.append(v)
		wr.writerow(ls)
