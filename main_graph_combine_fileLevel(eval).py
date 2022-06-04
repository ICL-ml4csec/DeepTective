import os
import pickle
import random
import torch
import activations as act
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

## Get gradients or not (IMPORTANT!). If set to TRUE, process will be slow. 
GET_GRAPH_GRADIENTS = False
GET_TOKENS_GRADIENTS = False

if GET_GRAPH_GRADIENTS == True or GET_TOKENS_GRADIENTS == True:
    TEST_BATCH_SIZE=1
else:
    TEST_BATCH_SIZE=32

## Define gradients related variable
weight_tokens, grad_tokens, ig_tokens, tokens, t_predicted, t_label, t_filenames = [], [], [], [], [], [], []
weight_graph, grad_graph, ig_graph, graph, graph_len, g_predicted, g_label, g_filenames = [], [], [], [], [], [], [], []

def _init_fn(worker_id):
    np.random.seed(seed + worker_id)

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

weight = torch.Tensor(np.array([1,24.1,4.9,30])).to(device=device)

dataset = 'custom'

## 'custom' for custom dataset

#################### LOAD DATA

f = open("data/CPG/%s_cfg_tokens_filelevel_3000.pkl" % dataset, 'rb')
data_all = np.array(pickle.load(f))
X_test = [[x_data[0], x_data[1], x_data[2], x_data[3]] for x_data in data_all]
        
print('evaluation data: ', len(X_test))



################## GRADIENTS EXTRACTION FUNCTION DEFINTION

def processing_tokens(model,pred,graph_data,token_data,embed_name,fname=None):

    lbl = graph_data.y.cpu().detach().numpy()[0]

    weight,grads,grad_whole_graph = act.get_weights_grads_from_model(model,pred,embed_name) ## added
    ig = act.integrated_gradients((graph_data,token_data),model,device,False,embed_name,50) ## added
    ig = ig[np.nonzero(ig)]
        
    output = pred.cpu().detach().numpy()
    output = np.argmax(output, axis=1)[0]

    weight_tokens.append(weight)
    grad_tokens.append(grads)
    ig_tokens.append(ig)
    tokens.append(token_data[0])
    t_predicted.append(output)
    t_label.append(lbl)
    t_filenames.append(fname)


def processing_graph(model,pred,graph_data,token_data,embed_name,fname=None):

    gph, gph_len = act.get_graph_tokens(graph_data.x)
    lbl = graph_data.y.cpu().detach().numpy()[0]

    weight,grads,grad_whole_graph = act.get_weights_grads_from_model(model,pred,embed_name) ## added
    ig = act.integrated_gradients((graph_data,token_data),model,device,True,embed_name,50) ## added
    ig = ig[np.nonzero(ig)]
        
    output = pred.cpu().detach().numpy()
    output = np.argmax(output, axis=1)[0]

    weight_graph.append(weight)
    grad_graph.append(grads)
    ig_graph.append(ig)
    graph.append(gph)
    graph_len.append(gph_len)
    g_predicted.append(output)
    g_label.append(lbl)
    g_filenames.append(fname)

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

    ys = []
    fname = []
    all_scores = np.empty([0,4])
    #with torch.no_grad():
    counter=0
    for t, x in enumerate(loader):
        print(counter,end='\r')
        #x1 = Batch.from_data_list(x).to(device)

        x_arr = np.array(x)
        x1 = x_arr[:, 0]
        x1 = Batch.from_data_list(x1).to(device)
        x2 = torch.stack(x_arr[:, 1].tolist(), dim=0).to(device=device, dtype=torch.long)
        
        scores = model(x1,x2,True)

        if GET_TOKENS_GRADIENTS is True:
            processing_tokens(model,scores,x1,x2,'embed',x_arr[:, 3])
        if GET_GRAPH_GRADIENTS is True:
            processing_graph(model,scores,x1,x2,'embed1',x_arr[:, 3])

        scores = nn.Softmax()(scores)
        vals = scores.cpu().detach().numpy()
        all_scores = np.concatenate((all_scores, vals), axis=0)
        preds = np.argmax(vals, axis=1)
        out.append(preds)
        y = torch.flatten(x1.y).cpu().detach().numpy()
        num_correct += np.where(preds == y,1,0).sum()
        num_samples += len(preds)
        for y1 in y:
            ys.append(y1)
        counter+=1
        fname = fname + [doc[3] for doc in x]
    acc = float(num_correct) / num_samples
    print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
    out = list(pd.core.common.flatten(out))
    pd.DataFrame({'fname':fname,
        'label':ys, 
        'predicted':out,
        'score_0': all_scores[:,0],
        'score_1': all_scores[:,1],
        'score_2': all_scores[:,2],
        'score_3': all_scores[:,3]}).to_csv('custom_eval.csv', index=False)

    return float(acc), out, ys

########################## Load the Model

model = PhpNetGraphTokensCombineFileLevel(VOCAB_SIZE_graph, VOCAB_SIZE_tokens)
model = torch.load("model_fileA.pt")
model.to(device)

######################### Perfrom prediction on new data

model.eval()

if GET_TOKENS_GRADIENTS is True or GET_GRAPH_GRADIENTS is True:
    act.set_eval_zerodrop(model)

print()
print(dataset)
my_dataloader = DataListLoader(X_test,batch_size=TEST_BATCH_SIZE,worker_init_fn=_init_fn, num_workers=0)
_, y_pred, y_true = check_accuracy(my_dataloader,model)
print(confusion_matrix(y_true,np.array(y_pred)))
print("precision")
print(precision_score(y_true,np.array(y_pred),average=None, labels=[0,1,2,3]))
print("recall")
print(recall_score(y_true,np.array(y_pred),average=None, labels=[0,1,2,3]))
print("f1")
print(f1_score(y_true,np.array(y_pred),average=None, labels=[0,1,2,3]))


####################### Process the gradients for TOKENS and GRAPHS

with open('data/CPG/mergeEncoder_cfg_filelevel.pkl','rb') as loader:
    merge_encoder = pickle.load(loader)

if GET_TOKENS_GRADIENTS is True:
    mydf_graph = act.process_activation(weight_tokens, 
                                  grad_tokens,
                                  ig_tokens,
                                  tokens,
                                  merge_encoder,
                                  t_predicted,
                                  t_label,
                                  t_filenames)

    mydf_graph.to_pickle('custom_tokens_grads.pkl')


if GET_GRAPH_GRADIENTS is True:
    mydf_graph = act.process_activation(weight_graph, 
                                  grad_graph,
                                  ig_graph,
                                  graph,
                                  merge_encoder,
                                  g_predicted,
                                  g_label,
                                  g_filenames,
                                  graph_len)

    mydf_graph.to_pickle('custom_graph_grads.pkl')
