from core import *
from sklearn.metrics import accuracy_score
from torch.utils.data import TensorDataset, DataLoader

import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import warnings


nonImg = NonImageToImage("Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
nonImg.convert2Image(isSave=False)


warnings.simplefilter('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
model = model.to(device)
preprocess = transforms.Compose([
    transforms.ToTensor()
])

X_train_tensor = torch.stack([preprocess(img) for img in nonImg.getTrain_all_images()]).float().to(device)
X_test_tensor = torch.stack([preprocess(img) for img in nonImg.getTest_all_features()]).float().to(device)

batch_size = 200
trainset = TensorDataset(X_train_tensor)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = TensorDataset(X_test_tensor)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    model.parameters(),
    lr=1e-04,
    momentum=0.8,
    weight_decay=1e-05
)

for epoch in range(200):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = device(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    if not (epoch % 20):
        print(f'[{epoch}] loss: {running_loss / len(X_train_tensor) * batch_size:.3f}')
print(f'[{epoch}] loss: {running_loss / len(X_train_tensor) * batch_size:.3f}')