from pyDeepInsight import Norm2Scaler
import image_transformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score
from torch.utils.data import TensorDataset, DataLoader

import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import warnings

file = "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
random_state=1515
csv = pd.read_csv(file, engine="c")

csv = csv.fillna(value=0)

for i in range(len(csv['Flow Bytes/s'])):
    if csv['Flow Bytes/s'][i]=='Infinity':
        csv['Flow Bytes/s'][i]= '1040000001'

for i in range(len(csv[' Flow Packets/s'])):
    if csv[' Flow Packets/s'][i]=='Infinity':
        csv[' Flow Packets/s'][i]= '2000001'

csv.replace([np.inf, -np.inf], np.nan, inplace=True)
rowsNan = list(csv[csv.isna().any(axis=1)].index)
csv.dropna(inplace=True)

featuresNotConvert = ['Flow ID', 'Source IP', 'Source Port', 'Destination IP', 'Destination Port', 'Protocol', 'Timestamp']
expert_features  = [c for c in csv.columns if c.strip() in featuresNotConvert]

for f in expert_features:
    csv = csv.drop(f, axis=1)

y = csv[' Label'].values
X = csv.iloc[:, :len(csv.columns)-1].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=23, stratify=y)

ln = Norm2Scaler()
X_train_norm = ln.fit_transform(X_train)
X_test_norm = ln.transform(X_test)

le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)
num_classes = np.unique(y_train_enc).size

distance_metric = 'cosine'
reducer = TSNE(
    n_components=2,
    metric=distance_metric,
    init='random',
    learning_rate='auto',
    n_jobs=-1
)

pixel_size = (227,227)
it = image_transformer.ImageTransformer(
    feature_extractor=reducer, 
    pixels=pixel_size)

it.fit(X_train, y=y_train, plot=False)
X_train_img = it.transform(X_train_norm, empty_value=0, format="rgb")
X_test_img = it.transform(X_test_norm)

# X_test_img = it.transform(X_test_norm)

print("done")

warnings.simplefilter('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
model = model.to(device)
preprocess = transforms.Compose([
    transforms.ToTensor()
])

X_train_tensor = torch.stack([preprocess(img) for img in X_train_img]).float().to(device)
y_train_tensor = torch.from_numpy(le.fit_transform(y_train)).to(device)

X_test_tensor = torch.stack([preprocess(img) for img in X_test_img]).float().to(device)
y_test_tensor = torch.from_numpy(le.transform(y_test)).to(device)

batch_size = 200

trainset = TensorDataset(X_train_tensor, y_train_tensor)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = TensorDataset(X_test_tensor, y_test_tensor)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

for epoch in range(1):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        # zero the parameter gradients
        print("[+] optimize")
        optimizer.zero_grad()
        # forward + backward + optimize
        print("[+] forward + backward + optimize")
        outputs = model(inputs)
        print("[+] criterion(outputs, labels)")
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    # print epoch statistics
    print(i)
    if not (epoch % 1):
        print(f'[{epoch}] loss: {running_loss / len(X_train_tensor) * batch_size:.3f}')
print(f'[{epoch}] loss: {running_loss / len(X_train_tensor) * batch_size:.3f}')

train_predicted = np.empty(0)
train_true = np.empty(0)
with torch.no_grad():
    model.eval()
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        pred = torch.max(model(inputs),1)[1].cpu().detach().numpy()
        train_predicted = np.append(train_predicted, pred)
        train_true = np.append(train_true, labels.cpu().detach().numpy())

test_predicted = np.empty(0)
test_true = np.empty(0)
with torch.no_grad():
    model.eval()
    for i, data in enumerate(testloader, 0):
        inputs, labels = data
        pred = torch.max(model(inputs),1)[1].cpu().detach().numpy()
        test_predicted = np.append(test_predicted, pred)
        test_true = np.append(test_true, labels.cpu().detach().numpy())

print(f"The train accuracy was {accuracy_score(train_predicted, train_true):.3f}")
print(f"The test accuracy was {accuracy_score(test_predicted, test_true):.3f}")