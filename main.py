import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score,recall_score,roc_curve,precision_score,f1_score
from sklearn.metrics import matthews_corrcoef
import matplotlib.pyplot as plt


def pretainPretrtain(file_path):
    df_esm = pd.read_csv(file_path)
    esm = df_esm.iloc[:,1:]
    return esm

def genData(file_path):
    data = pd.read_csv(file_path)
    x = data.iloc[:, 0]
    y = data.iloc[:, 1]
    return x, y

def finger(data):
    fingerprints = []
    finger = []
    for i in data.tolist():
        mol = Chem.MolFromFASTA(i)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=512)
        fingerprints.append(fp)
    for i in fingerprints:
        fp_str = i.ToBitString()
        fp_list = [int(bit) for bit in fp_str]
        finger.append(fp_list)
    fps= np.stack(finger)
    return fps

def concat(fps,esm,y):
    X = np.concatenate((fps, esm), axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.iloc[:,1], dtype=torch.float32).view(-1, 1)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.iloc[:,1].values, dtype=torch.float32).view(-1, 1)

    return X_train_tensor,y_train_tensor,X_test_tensor,y_test_tensor,y_test

def metrics(test, pred):
    Mcc = matthews_corrcoef(test, pred)
    f1_micro = f1_score(test, pred)
    Recall = recall_score(test, pred)
    ACC = accuracy_score(test, pred)
    pre = precision_score(test, pred)
    AUC = roc_auc_score(test, pred)

    return Mcc, f1_micro, Recall, ACC, pre, AUC


esm = pretainPretrtain('./dataset_esm_2_feature.csv')
x,y = genData('./data/dataset.csv')
fps = finger(x)
X_train_tensor,y_train_tensor,X_test_tensor,y_test_tensor,y_test =concat(fps,esm,y)

class CppNet(nn.Module):
    def __init__(self):
        super(CPP, self).__init__()
        # esm
        self.esm_lstm = nn.LSTM(1280, 32, 3, batch_first=True, bidirectional=True)
        self.esm_fc1 = nn.Linear(64, 32)

        self.bn_esm1 = nn.BatchNorm1d(64)
        self.bn_esm2 = nn.BatchNorm1d(32)
        self.esm_dropout1 = nn.Dropout(0.3)
        self.esm_dropout2 = nn.Dropout(0.2)

        # fps
        self.fp_conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=2)
        self.fp_pool = nn.MaxPool1d(kernel_size=2)
        self.fp_fc = nn.Linear(4112, 32)
        self.fp_bn1 = nn.BatchNorm1d(32)

        # out
        self.cpp_fc1 = nn.Linear(64, 32)
        self.cpp_fc2 = nn.Linear(32, 1)

    def forward(self, data):

        esm = data[:, 512:1792]
        fp = data[:, 0:512]
        # esm
        esm1, _ = self.esm_lstm(esm)
        esm2 = self.bn_esm2(self.esm_dropout2(self.esm_fc1(self.bn_esm1(self.esm_dropout1(esm1)))))
        # esm2 = self.bn_esm2(self.esm_dropout2(self.esm_fc1(esm)))

        # fps
        fp = F.relu(self.fp_conv1(fp.unsqueeze(1)))
        fp = self.fp_pool(fp)
        fp = fp.view(fp.size(0), -1)
        fp = self.fp_bn1(self.fp_fc(fp))

        # out
        out = torch.cat((fp, esm2), 1)
        x = F.sigmoid(self.cpp_fc2(self.cpp_fc1(out)))
        return x


CPP = CppNet()

# Define loss functions and optimizations
criterion = nn.BCELoss()
optimizer = optim.Adam(CPP.parameters(), lr=0.1)

# Learning Rate Scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

# Save the training loss and validation loss for each epoch
train_losses = []
num_epochs = 450

for epoch in range(num_epochs):
    CppNet.train()  # Set to training mode
    optimizer.zero_grad()
    outputs = CppNet(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    scheduler.step() # Update the learning rate
    train_losses.append(loss.item())  # Record training losses
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

plt.plot(train_losses, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.legend()
plt.title('Loss Function Curve')
plt.show()


CPP.eval()

with torch.no_grad():
    y_pred = CPP(X_test_tensor)

y_pred = y_pred.tolist()
y_pred1 = [item for sublist in y_pred for item in sublist]


pred = []
for i  in y_pred1:
    if i >0.5:
        pred.append(1)
    else:
        pred.append(0)

Mcc, f1_micro, Recall, ACC, pre, AUC = metrics(y_test,pred)


