import copy
import os
import pickle
import random
import time
import torch
import pandas as pd
import numpy as np
from model import PhpNetGraphTokensCombineFileLevel
from torch import nn
from torch_geometric.data import DataLoader, DataListLoader, Batch
from torch.utils import data
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score , precision_score, recall_score

seed = 42
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
VOCAB_SIZE_graph = 246347
VOCAB_SIZE_tokens = 246347

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

weight = torch.Tensor(np.array([1,24.1,4.9,30])).to(device=device)

#################### TRAIN TEST SPLIT

for dataset, val in {'sard':train_SARD, 'git':train_GIT}.items():
    f = open("data/CPG/%s_cfg_tokens_filelevel_3000.pkl" % dataset, 'rb')
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
        
print('Training data: ', len(X_train))


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

            #processing_graph(model,scores,x1,x2,'embed1')

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

my_dataloader = DataListLoader(X_train,batch_size=32,shuffle=True,worker_init_fn=_init_fn, num_workers=0)

model = PhpNetGraphTokensCombineFileLevel(VOCAB_SIZE_graph, VOCAB_SIZE_tokens)
#model = torch.load("model_fileA.pt")
model.to(device)
epochs=150
dtype = torch.long
print_every = 500
accs = []
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, min_lr=1e-6)

#########################  Train the model

tstart=time.perf_counter()
best_model = None
current_loss = 99999

for e in range(epochs):
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

        # Zero out all of the gradients for the variables which the optimizer
        # will update.
        loss.backward()

        # Update the parameters of the model using the gradients
        optimizer.step()

        if t % print_every == 0:
            #print('Epoch: %d, Iteration %d, loss = %.4f' % (e, t, loss.item()))
            scheduler.step(loss.item())
            #print()
        
    if loss.item() < current_loss:
        best_model = copy.deepcopy(model)
        current_loss = loss.item()

    print('Epoch: %d, loss = %.4f' % (e, loss.item()))
    
tstop=time.perf_counter()
print(tstop-tstart)

######################### Evaluate the current model

model.eval()
for name, X_test in testing_data.items():
    print()
    print(name)
    my_dataloader = DataListLoader(X_test,batch_size=32,worker_init_fn=_init_fn, num_workers=0)
    _, y_pred, y_true = check_accuracy(my_dataloader,model)
    y_pred = [element for sublist in y_pred for element in sublist]
    print(confusion_matrix(y_true,np.array(y_pred)))
    print("precision")
    print(precision_score(y_true,np.array(y_pred),average=None, labels=[0,1,2,3]))
    print("recall")
    print(recall_score(y_true,np.array(y_pred),average=None, labels=[0,1,2,3]))
    print("f1")
    print(f1_score(y_true,np.array(y_pred),average=None, labels=[0,1,2,3]))
    

print('\nsard + git')
my_dataloader = DataListLoader(testing_data['sard']+testing_data['git'],batch_size=32,worker_init_fn=_init_fn, num_workers=0)
_, y_pred, y_true = check_accuracy(my_dataloader,model)
y_pred = [element for sublist in y_pred for element in sublist]
print(confusion_matrix(y_true,np.array(y_pred)))
print("precision")
print(precision_score(y_true,np.array(y_pred),average=None, labels=[0,1,2,3]))
print("recall")
print(recall_score(y_true,np.array(y_pred),average=None, labels=[0,1,2,3]))
print("f1")
print(f1_score(y_true,np.array(y_pred),average=None, labels=[0,1,2,3]))

######################### Evaluate the best_model (model with lowest loss)

best_model.eval()
for name, X_test in testing_data.items():
    print()
    print(name)
    my_dataloader = DataListLoader(X_test,batch_size=32,worker_init_fn=_init_fn, num_workers=0)
    _, y_pred, y_true = check_accuracy(my_dataloader,best_model)
    y_pred = [element for sublist in y_pred for element in sublist]
    print(confusion_matrix(y_true,np.array(y_pred)))
    print("precision")
    print(precision_score(y_true,np.array(y_pred),average=None, labels=[0,1,2,3]))
    print("recall")
    print(recall_score(y_true,np.array(y_pred),average=None, labels=[0,1,2,3]))
    print("f1")
    print(f1_score(y_true,np.array(y_pred),average=None, labels=[0,1,2,3]))
    

print('\nsard + git')
my_dataloader = DataListLoader(testing_data['sard']+testing_data['git'],batch_size=32,worker_init_fn=_init_fn, num_workers=0)
_, y_pred, y_true = check_accuracy(my_dataloader,best_model)
y_pred = [element for sublist in y_pred for element in sublist]
print(confusion_matrix(y_true,np.array(y_pred)))
print("precision")
print(precision_score(y_true,np.array(y_pred),average=None, labels=[0,1,2,3]))
print("recall")
print(recall_score(y_true,np.array(y_pred),average=None, labels=[0,1,2,3]))
print("f1")
print(f1_score(y_true,np.array(y_pred),average=None, labels=[0,1,2,3]))


######################### Save models

torch.save(model, "model_cfg_tokens_fileLevel_3000_sard_git(current).pt")
torch.save(best_model, "model_cfg_tokens_fileLevel_3000_sard_git(best_model).pt")