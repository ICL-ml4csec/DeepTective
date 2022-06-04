import pickle
from model import PhpNet
import torch
from torch.utils import data
from torch import nn
import torch.nn.functional as F
from torch import optim
import pickle
from functools import lru_cache
import numpy as np
from sklearn.model_selection import train_test_split

# This file creates the TestSet for testing performance of other tools

f = open("data/sard_data_raw.pkl", 'rb')
data = np.array(pickle.load(f))
f = open("data/git_data_raw.pkl", 'rb')
data =  np.concatenate((data, np.array(pickle.load(f))), axis=0)
f = open("data/nvd_data_raw.pkl", 'rb')
data = np.concatenate((data, np.array(pickle.load(f))), axis=0)
print("data: "+len(data))

ys = [d[1] for d in data]
X_train, X_test = train_test_split(data, test_size=0.1,shuffle=True, stratify=ys) #42
file_no = 1
for func in X_test:
    name = "TestSet/test_" + str(func[1]) + "_" + str(file_no) +".php"
    with open(name, 'w') as f:
        f.write(func[0])
    file_no += 1
