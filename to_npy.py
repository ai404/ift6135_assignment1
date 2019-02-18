import glob
import os
import re

import cv2
import numpy as np
from tqdm import tqdm
import pickle

print(os.getcwd())

train_files = list(glob.iglob("./data/trainset/**/*.jpg"))  # [:10]
test_files = list(glob.iglob("./data/testset/**/*.jpg"))  # [:10]

Xtr = np.zeros((len(train_files), 64, 64, 3),dtype=np.uint8)
y = np.zeros((len(train_files), 2))

Xte = np.zeros((len(test_files), 64, 64, 3),dtype=np.uint8)

print(f"{len(train_files)} files found for training...")
for i, file in tqdm(enumerate(train_files)):
    matches = re.findall(r"/(\w*)\.(\w*)\.(\w*)", file)
    if len(matches):
        _, classe, ext = matches[0]
        assert classe in ["Cat", "Dog"]
        img = cv2.imread(file)
        Xtr[i, :] = img
        y[i, int(classe == "Dog")] = 1

print(f"{len(test_files)} files found for testing...")
test_ids = np.zeros(len(test_files))
for i, file in tqdm(enumerate(test_files)):
    matches = re.findall(r"/(\w*)\.(\w*)", file)
    if len(matches):
        id, ext = matches[0]
        test_ids[i] = id
        img = cv2.imread(file)
        Xte[i, :] = img

np.savez("data/data.npz", Xtr=Xtr, y=y, Xte=Xte,testIds = test_ids)
