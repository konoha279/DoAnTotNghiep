from pyDeepInsight import Norm2Scaler
import image_transformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score
from torch.utils.data import TensorDataset, DataLoader

import os, sys
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

warnings.simplefilter('ignore')

def handleCsv(csv):
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
    return csv

def convertImage(csv):
    y = csv[' Label'].values
    X = csv.iloc[:, :len(csv.columns)-1].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=23, stratify=y)

    ln = Norm2Scaler()
    X_train_norm = ln.fit_transform(X_train)
    X_test_norm = ln.transform(X_test)

    global le
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
    global it
    it = image_transformer.ImageTransformer(
        feature_extractor=reducer, 
        pixels=pixel_size)
    it.fit(X_train, y=y_train, plot=False)
    X_train_img = it.transform(X_train_norm, empty_value=0, format="rgb")
    X_test_img = it.transform(X_test_norm)
    # showImage_Test(it, X_train_img, y_train)
    return X_train_img, X_test_img, y_train, y_test

def preTrainData(X_train_img, X_test_img, y_train, y_test):
    X_train_tensor = torch.stack([preprocess(img) for img in X_train_img]).float().to(device)
    y_train_tensor = torch.from_numpy(le.fit_transform(y_train)).to(device)

    X_test_tensor = torch.stack([preprocess(img) for img in X_test_img]).float().to(device)
    y_test_tensor = torch.from_numpy(le.transform(y_test)).to(device)

    global batch_size
    batch_size = 200

    trainset = TensorDataset(X_train_tensor, y_train_tensor)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    testset = TensorDataset(X_test_tensor, y_test_tensor)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    return X_train_tensor, trainloader, testloader

def CNNTrainDataset(model, optimizer, criterion, X_train_tensor, trainloader, testloader, num_epochs=30):
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        # print epoch statistics
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
    return model

def showImage_Test(it, X_train_img, y_train):
    it.pixels = 50

    fdm = it.feature_density_matrix()
    fdm[fdm == 0] = np.nan

    plt.figure(figsize=(10, 7.5))

    cax = sns.heatmap(fdm, cmap="viridis", linewidths=0.01, 
                    linecolor="lightgrey", square=True)
    cax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    cax.yaxis.set_major_locator(ticker.MultipleLocator(5))
    for _, spine in cax.spines.items():
        spine.set_visible(True)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    for i in range(0,3):
        ax[i].imshow(X_train_img[i])
        ax[i].title.set_text(f"Train[{i}] - class '{y_train[i]}'")
    plt.tight_layout()
    plt.show()
    exit(0)

def main():
    FULLPATH = os.path.abspath(sys.argv[1])
    if not os.path.exists(FULLPATH):
        print("path input khong ton tai")
        return
    print("[+] read file csv")
    csv = pd.read_csv(FULLPATH, engine="c")

    print("[+] handle NaN and Infinity value in file csv")
    csv = handleCsv(csv)
    print("[+] convert dataset to images")
    X_train_img, X_test_img, y_train, y_test = convertImage(csv)

    print("[+] pre train data Image")
    X_train_tensor, trainloader, testloader = preTrainData(X_train_img, X_test_img, y_train, y_test)

    print("[+] create resnet18")
    model_resnet18 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    model_resnet18 = model_resnet18.to(device)
    criterion_resnet18  = nn.CrossEntropyLoss()
    optimizer_resnet18  = optim.SGD(model_resnet18.parameters(), lr=0.001, momentum=0.9)
    print("[+] resnet18 training data")
    model_resnet18 = CNNTrainDataset(model_resnet18, 
                                optimizer_resnet18, 
                                criterion_resnet18, 
                                X_train_tensor, 
                                trainloader, 
                                testloader, 
                                num_epochs=30)
    
    print("[+] create resnet34")
    model_resnet34 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True)
    model_resnet34 = model_resnet34.to(device)
    criterion_resnet34  = nn.CrossEntropyLoss()
    optimizer_resnet34  = optim.SGD(model_resnet34.parameters(), lr=0.001, momentum=0.9)
    print("[+] resnet34 training data")
    model_resnet34 = CNNTrainDataset(model_resnet34, 
                                optimizer_resnet34, 
                                criterion_resnet34, 
                                X_train_tensor, 
                                trainloader, 
                                testloader, 
                                num_epochs=30)

    print("[+] create vgg")
    model_vgg = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11', pretrained=True)
    model_vgg = model_vgg.to(device)
    criterion_vgg  = nn.CrossEntropyLoss()
    optimizer_vgg  = optim.SGD(model_vgg.parameters(), lr=0.001, momentum=0.9)
    print("[+] vgg training data")
    model_vgg = CNNTrainDataset(model_vgg, 
                            optimizer_vgg, 
                            criterion_vgg, 
                            X_train_tensor, 
                            trainloader, 
                            testloader, 
                            num_epochs=30)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("main.py <folder datasets>")
        exit(0)
    global preprocess, device
    preprocess = transforms.Compose([
        transforms.ToTensor()
    ])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    main()