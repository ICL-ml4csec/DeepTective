from model import PhpNetGraphTokensCombine
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
from data.preprocessing import sub_tokens
from torch_geometric.data import DataLoader, DataListLoader, Batch
from torch.utils import data
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score , precision_score, recall_score
import main
import config
import random
import os

#ignore warnings for scikit learn functions when labels do not have any values
import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)


# Set the seed
seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

weight = torch.Tensor(np.array([1,24.1,4.9,30])).to(device=device) #[1,24.1,4.9,30]

if config.train_on_sard or config.test_on_sard:
    if config.sard_with_no_duplicates:
        f_graph = open("data/CFG/sard_multi_replace_tokens_graph.pkl", 'rb')
        f_tokens = open("data/Tokens/sard_multi_replace_tokens_no_dup.pkl", 'rb')
    else:
        f_graph = open("data/CFG/sard_multi_replace_tokens_graph_with_dup.pkl", 'rb')
        f_tokens = open("data/Tokens/sard_multi_replace_tokens_with_dup.pkl", 'rb')
    data_sard_graph = np.array(pickle.load(f_graph))
    data_sard_tokens = np.array(pickle.load(f_tokens))

if config.train_on_git or config.test_on_git:
    f_graph = open("data/CFG/git_graph_tokens_no_dup.pkl", 'rb')
    f_tokens = open("data/Tokens/git_replaced_tokens_no_dup.pkl", 'rb')
    data_git_graph = np.array(pickle.load(f_graph))
    data_git_tokens = np.array(pickle.load(f_tokens))

if config.train_on_nvd or config.test_on_nvd:
    f_graph = open("data/CFG/nvd_graph_tokens_no_dup.pkl", 'rb')
    f_tokens = open("data/Tokens/nvd_replace_tokens_no_dup.pkl", 'rb')
    data_nvd_graph = np.array(pickle.load(f_graph))
    data_nvd_tokens = np.array(pickle.load(f_tokens))


# print(data_sard.shape)

def _init_fn(worker_id):
    np.random.seed(seed + worker_id)

def get_data_custom(data_graph, data_tokens):
    # 1 here as wont load as Data otherwise
    x,y = main.get_data_custom(data_tokens)
    xs = [[x11[0],da] for (x11,da) in zip(data_graph,x)]
    ys = [dat[0].y for dat in data_graph[:, :1]]
    return xs,ys

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
        for t, x in enumerate(loader):
            x_arr = np.array(x)
            x1 = x_arr[:, 0]
            x1 = Batch.from_data_list(x1).to(device)
            x2 = torch.stack(x_arr[:, 1].tolist(), dim=0).to(device=device, dtype=torch.long)
            scores = model(x1,x2)
            vals = scores.cpu().detach().numpy()
            preds = np.argmax(vals, axis=1)
            out.append(preds)
            y = torch.flatten(x1.y).cpu().detach().numpy()
            num_correct += np.where(preds == y,1,0).sum()
            num_samples += len(preds)
            for y1 in y:
                ys.append(y1)
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
        return float(acc), out, ys

x_SARD_train = []
if config.train_on_sard or config.test_on_sard:
    x_SARD, y_SARD = get_data_custom(data_sard_graph, data_sard_tokens)
    if not config.test_on_sard:
        x_SARD_train = x_SARD
    elif not config.train_on_sard:
        x_SARD_train = []
        x_SARD_test = x_SARD
    else:
        x_SARD_train, x_SARD_test, _, y_SARD_test = train_test_split(x_SARD,y_SARD, test_size=0.1, shuffle=True,stratify=y_SARD, random_state=seed)

x_GIT_train = []
if config.train_on_git or config.test_on_git:
    x_GIT, y_GIT = get_data_custom(data_git_graph, data_git_tokens)
    if not config.test_on_git:
        x_GIT_train = x_GIT
    elif not config.train_on_git:
        x_GIT_train = []
        x_GIT_test = x_GIT
    else:
        x_GIT_train, x_GIT_test, _, y_GIT_test = train_test_split(x_GIT, y_GIT, test_size=0.1, shuffle=True,
                                                                  stratify=y_GIT, random_state=seed)

x_NVD_train = []
if config.train_on_nvd or config.test_on_nvd:
    x_NVD, y_NVD = get_data_custom(data_nvd_graph, data_nvd_tokens)
    if not config.test_on_nvd:
        x_NVD_train = x_NVD
    elif not config.train_on_nvd:
        x_NVD_test = x_NVD
    else:
        x_NVD_train, x_NVD_test, _, y_NVD_test = train_test_split(x_NVD,y_NVD, test_size=0.1, shuffle=True,stratify=y_NVD, random_state=seed)

X_train = x_SARD_train + x_GIT_train + x_NVD_train


#X_train, X_val = train_test_split(X_train, test_size=1/9)
my_dataloader = DataListLoader(X_train,batch_size=64,shuffle=True,worker_init_fn=_init_fn, num_workers=0)
model = PhpNetGraphTokensCombine()
model.to(device)
epochs=150
dtype = torch.long
print_every = 500
accs = []
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001) #optim.Adagrad(model.parameters(),lr=0.001, weight_decay=0.00001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, min_lr=1e-6)
for e in range(epochs):
        for t, x in enumerate(my_dataloader):
            x_arr = np.array(x)
            x1 = x_arr[:,0]
            x1 = Batch.from_data_list(x1).to(device)
            x2 = torch.stack(x_arr[:, 1].tolist(), dim=0).to(device=device, dtype=torch.long)
            model.train()  # put model to training mode
            optimizer.zero_grad()
            outputs = model(x1,x2)
            y = x1.y
            loss = nn.CrossEntropyLoss(weight=weight)(outputs, y)
            # Zero out all of the gradients for the variables which the optimizer
            # will update.

            loss.backward()

            # Update the parameters of the model using the gradients
            optimizer.step()

            if t % print_every == 0:
                print('Epoch: %d, Iteration %d, loss = %.4f' % (e, t, loss.item()))
                scheduler.step(loss.item())
                print()


# torch.save(model, "model_combine.pt")
model.eval()
if config.test_on_sard:
    my_dataloader = DataListLoader(x_SARD_test,batch_size=32,worker_init_fn=_init_fn, num_workers=0)
    _, y_pred, y_true = check_accuracy(my_dataloader,model)
    y_pred = [element for sublist in y_pred for element in sublist]
    print("SARD")
    print(confusion_matrix(y_true,np.array(y_pred),labels=[0,1,2,3]))
    print("precision")
    print(precision_score(y_true,np.array(y_pred),average=None, labels=[0,1,2,3]))
    print("recall")
    print(recall_score(y_true,np.array(y_pred),average=None, labels=[0,1,2,3]))
    print("f1")
    print(f1_score(y_true,np.array(y_pred),average=None, labels=[0,1,2,3]))

if config.test_on_git:
    my_dataloader = DataListLoader(x_GIT_test,batch_size=32,worker_init_fn=_init_fn, num_workers=0)
    _, y_pred, y_true = check_accuracy(my_dataloader,model)
    y_pred = [element for sublist in y_pred for element in sublist]
    print("GIT")
    print(confusion_matrix(y_true,np.array(y_pred),labels=[0,1,2,3]))
    print("precision")
    print(precision_score(y_true,np.array(y_pred),average=None, labels=[0,1,2,3]))
    print("recall")
    print(recall_score(y_true,np.array(y_pred),average=None, labels=[0,1,2,3]))
    print("f1")
    print(f1_score(y_true,np.array(y_pred),average=None, labels=[0,1,2,3]))

if config.test_on_nvd:
    my_dataloader = DataListLoader(x_NVD_test,batch_size=32,worker_init_fn=_init_fn, num_workers=0)
    _, y_pred, y_true = check_accuracy(my_dataloader,model)
    y_pred = [element for sublist in y_pred for element in sublist]
    print("NVD")
    print(confusion_matrix(y_true,np.array(y_pred),labels=[0,1,2,3]))
    print("precision")
    print(precision_score(y_true,np.array(y_pred),average=None, labels=[0,1,2,3]))
    print("recall")
    print(recall_score(y_true,np.array(y_pred),average=None, labels=[0,1,2,3]))
    print("f1")
    print(f1_score(y_true,np.array(y_pred),average=None, labels=[0,1,2,3]))
