from model import PhpNetW2V
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

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

weight = torch.Tensor(np.array([1.5,15,3.3,21.4])).to(device=device)

f = open("data/all_multi_replace_w2v.pkl", 'rb')
data_sard_w2v = np.array(pickle.load(f))

f = open("data/sard_multi_replace_tokens_no_dup.pkl", 'rb')
data_sard = np.array(pickle.load(f))
print(len(data_sard))
f = open("data/git_replaced_tokens_no_dup.pkl", 'rb')
data_sard =  np.concatenate((data_sard, np.array(pickle.load(f))), axis=0)
f = open("data/nvd_replace_tokens_no_dup.pkl", 'rb')
data_sard = np.concatenate((data_sard, np.array(pickle.load(f))), axis=0)
print(data_sard.shape)

print(data_sard_w2v.shape)


@lru_cache(maxsize=32)
def get_data():

    x = data_sard_w2v

    max_len = 200

    temp_x = np.zeros((len(x),max_len,300))
    i = 0
    for arr in x:
        temp_x[i,(-len(arr)):] = arr[:max_len]
        i += 1
    x = torch.tensor(temp_x)
    y = [item for sublist in data_sard[:, 1:] for item in sublist]
    y = torch.tensor(y)
    print(max_len)
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
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
        return float(acc), out

x,y = get_data()
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1) #42
my_dataset = data.TensorDataset(X_train,y_train)
my_dataloader = data.DataLoader(my_dataset,batch_size=512,shuffle=True)

model = PhpNetW2V(300,100,2,True,0.5,4)
model.to(device)
epochs=50
dtype = torch.float
print_every = 50
optimizer = optim.Adam(model.parameters(),lr=0.0001)
for e in range(epochs):
        for t, (x, y) in enumerate(my_dataloader):
            model.train()  # put model to training mode
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)
            outputs = model(x)
            criterion = nn.CrossEntropyLoss(weight=weight)
            loss = criterion(outputs,y)
            optimizer.zero_grad()

            loss.backward()

            # Update the parameters of the model using the gradients
            optimizer.step()

            if t % print_every == 0:
                print('Epoch: %d, Iteration %d, loss = %.4f' % (e, t, loss.item()))
                print()


my_dataset = data.TensorDataset(X_test.to(device=device),y_test.to(device=device))
my_dataloader = data.DataLoader(my_dataset,batch_size=32)
_, y_pred = check_accuracy(my_dataloader,model)
y_pred = [element for sublist in y_pred for element in sublist]
print(confusion_matrix(y_test.numpy(),np.array(y_pred)))
