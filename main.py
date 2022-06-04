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
from sklearn.metrics import confusion_matrix
from torchnlp.encoders import LabelEncoder
import matplotlib.pyplot as plt
from data.preprocessing import sub_tokens, map_tokens
from sklearn.metrics import confusion_matrix, f1_score , precision_score, recall_score
import util
import config
# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

weight = torch.Tensor(np.array([1,24.1,4.9,30])).to(device=device) #[1,15,3.3,21.4]  #[9.6, 1.9, 12.6, 1, 1.66, 5.2] [1,8.7,5.1,11.1][1,8,4.9,30]

# if config.train_on_sard:
#     f = open("data/Tokens/sard_multi_replace_tokens_with_dup.pkl", 'rb')
#     data_sard = np.array(pickle.load(f))
# else:
#     f = open("data/Tokens/sard_multi_replace_tokens_no_dup.pkl", 'rb')
#     data_sard = np.array(pickle.load(f))
#     f = open("data/Tokens/git_replaced_tokens_no_dup.pkl", 'rb')
#     data_sard =  np.concatenate((data_sard, np.array(pickle.load(f))), axis=0)
#     f = open("data/Tokens/nvd_replace_tokens_no_dup.pkl", 'rb')
#     data_sard = np.concatenate((data_sard, np.array(pickle.load(f))), axis=0)


tokens = map_tokens.tokens

def get_data_custom_no_y(data_in):
    encoder = LabelEncoder(tokens)
    x = []
    for lines in data_in:
        x_curr = []
        for token in lines:
            enc = encoder.encode(token)
            if enc == 0 :
                print("error")
                print(token)
                exit(1)
            x_curr.append(enc)
        x.append(x_curr)

    max_len = 0
    for arr in x:
        max_len = len(arr) if len(arr) > max_len else max_len

    temp_x = np.zeros((len(x),max_len))
    i = 0
    for arr in x:
        temp_x[i,:len(arr)] = arr
        i += 1
    x = torch.tensor(temp_x)
    return x

def get_data_custom(data_in):
    encoder = LabelEncoder(tokens)

    x = []
    for lines in data_in[:, :1]:
        x_curr = []
        for token in lines[0]:
            enc = encoder.encode(token)
            if enc == 0:
                print("error")
                print(token)
                exit(1)
            x_curr.append(enc)
        x.append(x_curr)

    max_len = 200

    temp_x = np.zeros((len(x),max_len))
    i = 0
    for arr in x:
        temp_x[i,(-len(arr)):] = arr[:max_len]
        i += 1
    x = torch.tensor(temp_x)
    y = [item for sublist in data_in[:, 1:] for item in sublist]
    y = torch.tensor(y)
    return x,y


@lru_cache(maxsize=32)
def get_data():
    encoder = LabelEncoder(tokens)

    x = []
    for lines in data_sard[:, :1]:
        x_curr = []
        for token in lines[0]:
            enc = encoder.encode(token)
            if enc == 0:
                print("error")
                print(token)
                exit(1)
            x_curr.append(enc)
        x.append(x_curr)

    max_len = 200

    temp_x = np.zeros((len(x),max_len))
    i = 0
    for arr in x:
        temp_x[i,(-len(arr)):] = arr[:max_len]
        i += 1
    x = torch.tensor(temp_x)
    y = [item for sublist in data_sard[:, 1:] for item in sublist]
    y = torch.tensor(y)
    return x,y

def check_accuracy(loader, model):
    # function for test accuracy on validation and test set
    out = []
    if False:  # loader.dataset.train:
        print('Checking accuracy on validation set')
    else:
        print('Checking accuracy on test set')
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    ys = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=dtype)  # move to device1
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            vals = scores.cpu().detach().numpy()
            preds = np.argmax(vals, axis=1)
            out.append(preds)
            y = torch.flatten(y).cpu().detach().numpy()
            num_correct += np.where(preds == y,1,0).sum()
            num_samples += len(preds)
            for y1 in y:
                ys.append(y1)
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
        return float(acc), out, ys
# 
# x,y = get_data()
# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1,shuffle=True, stratify=y)
# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=1/9)
# my_dataset = data.TensorDataset(X_train,y_train)
# my_dataloader = data.DataLoader(my_dataset,batch_size=128,shuffle=True)
#
# model = PhpNet(5000,200,200,3,True,0.5,4)
# model.to(device)
# epochs=100
# dtype = torch.long
# print_every = 500
# optimizer = optim.Adam(model.parameters(),lr=0.0001)
# accs= []
# losses = []
# accs_train = []
# accs_val = []
# for e in range(epochs):
#         for t, (x, y) in enumerate(my_dataloader):
#             model.train()  # put model to training mode
#             x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
#             y = y.to(device=device, dtype=torch.long)
#             outputs = model(x)
#             criterion = nn.CrossEntropyLoss(weight=weight)
#             loss = criterion(outputs,y)
#             optimizer.zero_grad()
#
#             loss.backward()
#
#             optimizer.step()
#
#             if t % print_every == 0:
#                 print('Epoch: %d, Iteration %d, loss = %.4f' % (e, t, loss.item()))
#                 losses.append(loss.item())
#                 print()
#
# my_dataset = data.TensorDataset(X_test,y_test)
# my_dataloader = data.DataLoader(my_dataset,batch_size=256)
# _, y_pred, y_true = check_accuracy(my_dataloader,model)
# y_pred = [element for sublist in y_pred for element in sublist]
# print(confusion_matrix(y_true,np.array(y_pred)))
# util.plot_confusion_matrix(y_pred,y_true,["Safe", "SQLi", "XSS", "CI"],normalize='true')
# util.plot_confusion_matrix(y_pred,y_true,["Safe", "SQLi", "XSS", "CI"],values_format="d")
# plt.show()
# print("precision")
# print(precision_score(y_true,np.array(y_pred),average=None, labels=[0,1,2,3]))
# print("recall")
# print(recall_score(y_true,np.array(y_pred),average=None, labels=[0,1,2,3]))
# print("f1")
# print(f1_score(y_true,np.array(y_pred),average=None, labels=[0,1,2,3]))
#
#
# my_dataset = data.TensorDataset(X_test,y_test)
# my_dataloader = data.DataLoader(my_dataset,batch_size=256)
# _, y_pred, y_true = check_accuracy(my_dataloader,model)
# y_pred = [element for sublist in y_pred for element in sublist]
# print(confusion_matrix(y_true,np.array(y_pred)))
