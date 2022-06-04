from model import PhpNetGraphTokensAST
import torch
from torch.utils import data
from torch import nn
import torch.nn.functional as F
from torch import optim
import pickle
from functools import lru_cache
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score , precision_score, recall_score
from torchnlp.encoders import LabelEncoder
from data.preprocessing import sub_tokens
from torch_geometric.data import DataLoader, DataListLoader, Batch
from torch.utils import data
import matplotlib.pyplot as plt
# import main
import util

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

weight = torch.Tensor(np.array([1,24.1,4.9,30])).to(device=device) #[1,15,3.3,21.4] [1,8.9,4.4,11.2]

f = open("data/AST/sard_multi_replace_tokens_AST_token.pkl", 'rb')
data_sard = np.array(pickle.load(f))
print(len(data_sard))
f = open("data/AST/git_AST_tokens_no_dup.pkl", 'rb')
data_sard = np.concatenate((data_sard, np.array(pickle.load(f))), axis=0)
f = open("data/AST/nvd_AST_tokens_no_dup.pkl", 'rb')
data_sard = np.concatenate((data_sard, np.array(pickle.load(f))), axis=0)
print(data_sard.shape)

@lru_cache(maxsize=32)
def get_data():
    xs = [[dat[0],1] for dat in data_sard[:, :1]]
    ys = [dat[0].y for dat in data_sard[:, :1]]
    return xs, ys

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
            # x2 = torch.stack(x_arr[:, 1].tolist(), dim=0).to(device=device, dtype=torch.long)
            scores = model(x1)
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

x,y = get_data()
X_train, X_test = train_test_split(x, test_size=0.1,stratify=y) #42
X_train, X_val = train_test_split(X_train, test_size=1/9)
my_dataloader = DataListLoader(X_train,batch_size=64,shuffle=True)
model = PhpNetGraphTokensAST()
model.to(device)
epochs=100
dtype = torch.long
print_every = 500
losses = []
accs_train = []
accs_val = []

optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True)
for e in range(epochs):
        for t, x in enumerate(my_dataloader):
            x_arr = np.array(x)
            x1 = x_arr[:,0]
            x1 = Batch.from_data_list(x1).to(device)
            model.train()  # put model to training mode
            outputs = model(x1)
            loss = nn.CrossEntropyLoss(weight=weight)(outputs, x1.y)

            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            optimizer.zero_grad()

            loss.backward()

            # Update the parameters of the model using the gradients
            optimizer.step()

            if t % print_every == 0:
                print('Epoch: %d, Iteration %d, loss = %.4f' % (e, t, loss.item()))
                scheduler.step(loss.item())
                print()

my_dataloader = DataListLoader(X_test,batch_size=128)
_, y_pred, y_true = check_accuracy(my_dataloader,model)
y_pred = [element for sublist in y_pred for element in sublist]
print(confusion_matrix(y_true,np.array(y_pred)))
util.plot_confusion_matrix(y_pred,y_true,["Safe", "SQLi", "XSS", "CI"],normalize='true')
util.plot_confusion_matrix(y_pred,y_true,["Safe", "SQLi", "XSS", "CI"],values_format="d")
plt.show()
print("precision")
print(precision_score(y_true,np.array(y_pred),average=None, labels=[0,1,2,3]))
print("recall")
print(recall_score(y_true,np.array(y_pred),average=None, labels=[0,1,2,3]))
print("f1")
print(f1_score(y_true,np.array(y_pred),average=None, labels=[0,1,2,3]))
