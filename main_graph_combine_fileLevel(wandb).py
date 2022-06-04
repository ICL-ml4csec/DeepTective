import copy
import os
import pickle
import random
import time
import torch
import wandb
import pandas as pd
import numpy as np
from model import PhpNetGraphTokensCombineFileLevel
from torch import nn
from torch_geometric.data import DataLoader, DataListLoader, Batch
from torch.utils import data
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score , precision_score, recall_score, accuracy_score

# Init wandb
wandb.init(project="deeptective", entity="ml-vuln")
wandb_config = wandb.config 
wandb_config.seed = 42
wandb_config.batch_size = 32
wandb_config.test_batch_size = 32
wandb_config.epochs = 100
wandb_config.lr = 0.00001
wandb_config.vocabsize_graph = 246347
wandb_config.vocabsize_tokens = 246347
wandb_config.max_node_length = 20
wandb_config.max_token_length = 3000
wandb_config.granularity = 'file'
wandb_config.model_name = 'File-A'


seed = wandb_config.seed
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

## Check the vocabulary size from the mergeEncoder.pkl file. Must be the same.

X_train = []
testing_data = {}

## Choose dataset for training
train_SARD = True
train_GIT  = True

## Choose dataset for testing
test_SARD = True
test_GIT  = True

def _init_fn(worker_id):
    np.random.seed(seed + worker_id)

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

#################### TRAIN TEST SPLIT

for dataset, val in {'sard':train_SARD, 'git':train_GIT}.items():
    filepath = "data/CPG/%s_cfg_tokens_filelevel_3000.pkl" % dataset
    f = open(filepath, 'rb')
    data_all = np.array(pickle.load(f))
    x = [[x_data[0], x_data[1], x_data[2], x_data[3]] for x_data in data_all]
    y = [x_data[0].y for x_data in data_all]
    X_train_temp, X_test_temp,  _, y_test_temp = train_test_split(x,
                                                   y,
                                                   test_size=0.1, 
                                                   shuffle=True,
                                                   stratify=y, 
                                                   random_state=seed)
    if val is True:
        X_train.extend(X_train_temp)
        if {'sard':test_SARD,'git':test_GIT}[dataset] is True:
            testing_data[dataset] = X_test_temp
    else:
        if {'sard':test_SARD,'git':test_GIT}[dataset] is True:
            testing_data[dataset] = x
            
#     artifact_name = "%s-filelevel-3000tokens-20nodes" % dataset
#     artifact = wandb.Artifact(artifact_name, type='dataset')
#     artifact.add_file(filepath)
#     wandb.run.log_artifact(artifact)
        
print('Training data: ', len(X_train))

################### WEIGHTS

y_train = [doc[0].y for doc in X_train]
wandb_config.class_weights = compute_class_weight('balanced',[0,1,2,3],y_train).tolist()
weight = torch.Tensor(np.array(wandb_config.class_weights)).to(device=device)

################### CHECK ACCURACY FUNCTION DEFINITION

def check_accuracy(loader, model):
    # function for test accuracy on validation and test set
    out = []
    if False:  # loader.dataset.train:
        print('Checking accuracy on validation set')
    else:
        print('Checking accuracy on test set')
    num_correct = 0
    num_samples = 0
    #model.eval()  # set model to evaluation mode
    ys = []
    with torch.no_grad():
        counter=0
        for t, x in enumerate(loader):
            print(counter,end='\r')
            #x1 = Batch.from_data_list(x).to(device)

            x_arr = np.array(x)
            x1 = x_arr[:, 0]
            x1 = Batch.from_data_list(x1).to(device)
            x2 = torch.stack(x_arr[:, 1].tolist(), dim=0).to(device=device, dtype=torch.long)
            
            scores = model(x1,x2)
            scores = nn.Softmax()(scores)
            vals = scores.cpu().detach().numpy()
            preds = np.argmax(vals, axis=1)
            out.append(preds)
            y = torch.flatten(x1.y).cpu().detach().numpy()
            num_correct += np.where(preds == y,1,0).sum()
            num_samples += len(preds)
            for y1 in y:
                ys.append(y1)
            counter+=1
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
        return float(acc), out, ys

########################## Load the data and Model

my_dataloader = DataListLoader(X_train,batch_size=wandb_config.batch_size,shuffle=True,worker_init_fn=_init_fn, num_workers=0)

model = PhpNetGraphTokensCombineFileLevel(wandb_config.vocabsize_graph, wandb_config.vocabsize_tokens)
#model = torch.load("data/zz/model_cfg_tokens_fileLevel_3000_sard_git.pt")
model.to(device)
dtype = torch.long
accs = []
optimizer = torch.optim.Adam(model.parameters(), lr=wandb_config.lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, min_lr=1e-6)
wandb.watch(model, log="all")

#########################  Train the model

tstart=time.perf_counter()

for e in range(wandb_config.epochs):
    for t, x in enumerate(my_dataloader):
        
        x_arr = np.array(x)
        x1 = x_arr[:,0]
        x1 = Batch.from_data_list(x1).to(device)
        x2 = torch.stack(x_arr[:, 1].tolist(), dim=0).to(device=device, dtype=torch.long)
        model.train()
        optimizer.zero_grad()
        outputs = model(x1,x2)
    
        y = x1.y
        loss = nn.CrossEntropyLoss(weight=weight)(outputs, y)
        loss.backward()
        optimizer.step()
        
    scheduler.step(loss.item())
    wandb.log({'epoch': e, 'loss': loss.item()})
    
tstop=time.perf_counter()
print(tstop-tstart)

######################### Evaluate the current model

torch.save(model.state_dict(), "model_%s_checkpoint.h5" % wandb_config.model_name)
wandb.save("model_%s_checkpoint.h5" % wandb_config.model_name)


model.eval()
for name, X_test in testing_data.items():
    all_ypred = []
    all_ytrue = []
    print()
    print(name)
    my_dataloader = DataListLoader(X_test,batch_size=wandb_config.test_batch_size,worker_init_fn=_init_fn, num_workers=0)
    _, y_pred, y_true = check_accuracy(my_dataloader,model)
    y_pred = [element for sublist in y_pred for element in sublist]
    print(confusion_matrix(y_true,np.array(y_pred)))
    print("precision")
    print(precision_score(y_true,np.array(y_pred),average=None, labels=[0,1,2,3]))
    print("recall")
    print(recall_score(y_true,np.array(y_pred),average=None, labels=[0,1,2,3]))
    print("f1")
    print(f1_score(y_true,np.array(y_pred),average=None, labels=[0,1,2,3]))
    
    all_ytrue = all_ytrue + y_true
    all_ypred = all_ypred + np.array(y_pred).tolist()
    all_ytrue = [1 if e > 0 else 0 for e in all_ytrue]
    all_ypred = [1 if e > 0 else 0 for e in all_ypred]
    print('binary confusion matrix')
    print(confusion_matrix(all_ytrue,all_ypred))
    print("binary precision")
    print(precision_score(all_ytrue,all_ypred))
    print("binary recall")
    print(recall_score(all_ytrue,all_ypred))
    print("binary f1")
    print(f1_score(all_ytrue,all_ypred))
    
all_ypred = []
all_ytrue = []
print('\nsard + git')
my_dataloader = DataListLoader(testing_data['sard']+testing_data['git'],batch_size=wandb_config.test_batch_size,worker_init_fn=_init_fn, num_workers=0)
_, y_pred, y_true = check_accuracy(my_dataloader,model)
y_pred = [element for sublist in y_pred for element in sublist]
print(confusion_matrix(y_true,np.array(y_pred)))
print("precision")
print(precision_score(y_true,np.array(y_pred),average=None, labels=[0,1,2,3]))
print("recall")
print(recall_score(y_true,np.array(y_pred),average=None, labels=[0,1,2,3]))
print("f1")
print(f1_score(y_true,np.array(y_pred),average=None, labels=[0,1,2,3]))
all_ytrue = all_ytrue + y_true
all_ypred = all_ypred + np.array(y_pred).tolist()

all_ytrue = [1 if e > 0 else 0 for e in all_ytrue]
all_ypred = [1 if e > 0 else 0 for e in all_ypred]

print('binary confusion matrix')
print(confusion_matrix(all_ytrue,all_ypred))
print("binary precision")
print(precision_score(all_ytrue,all_ypred))
print("binary recall")
print(recall_score(all_ytrue,all_ypred))
print("binary f1")
print(f1_score(all_ytrue,all_ypred))

wandb.sklearn.plot_confusion_matrix(all_ytrue, all_ypred, labels=['Safe','Unsafe'])
wandb.log({'Accuracy': accuracy_score(all_ytrue, all_ypred), 
           'Precision': precision_score(all_ytrue, all_ypred), 
           'Recall': recall_score(all_ytrue, all_ypred), 
           'F1':f1_score(all_ytrue, all_ypred)})
