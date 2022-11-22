from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import image_transformer
import os

def plot_embed_2D(X, title=None):
    sns.set(style="darkgrid")

    # Create subplots
    fig, ax = plt.subplots(1, 1, figsize=(10, 7), squeeze=False)
    ax[0, 0].scatter(X[:, 0],
                     X[:, 1],
                     cmap=plt.cm.get_cmap("jet", 10),
                     marker="x",
                     alpha=1.0)
    plt.gca().set_aspect('equal', adjustable='box')

    if title is not None:
        ax[0, 0].set_title(title, fontsize=20)

    plt.rcParams.update({'font.size': 14})
    plt.show()

def tsne_transform(data, perplexity=30, plot=True):
    # Transpose to get (n_features, n_samples)
    data = data.T

    tsne = TSNE(n_components=2,
                metric='cosine',
                perplexity=perplexity,
                n_iter=1000,
                method='exact',
                random_state=10,
                n_jobs=-1)
    # Transpose to get (n_features, n_samples)
    transformed = tsne.fit_transform(data)

    if plot:
        plot_embed_2D(
            transformed,
            f"All Feature Location Matrix of Training Set (Perplexity: {perplexity})"
        )
    return transformed

def plot_feature_density(it, pixels=100, show_grid=True, title=None, isShow=False, folderSaving="outputImage"):
    # Update image size
    it.pixels = pixels

    fdm = it.feature_density_matrix()
    fdm[fdm == 0] = np.nan

    # Create subplots
    fig, ax = plt.subplots(1, 1, figsize=(10, 7), squeeze=False)

    if show_grid:
        sns.heatmap(fdm,
                    cmap="viridis",
                    linewidths=0.01,
                    linecolor="lightgrey",
                    square=False,
                    ax=ax[0, 0])
        for _, spine in ax[0, 0].spines.items():
            spine.set_visible(True)
    else:
        sns.heatmap(fdm,
                    cmap="viridis",
                    linewidths=0,
                    square=False,
                    ax=ax[0, 0])

    if title is not None:
        ax[0, 0].set_title(title, fontsize=20)

    plt.rcParams.update({'font.size': 14})
    print("[+] save All_Feature_Density_Matrix_of_Training_Set.png")
    plt.savefig(folderSaving + "/All_Feature_Density_Matrix_of_Training_Set.png")
    if isShow:
        plt.show()
    plt.clf()
    # Feature Overlapping Counts
    gene_overlap = (
        pd.DataFrame(it._coords.T).assign(count=1).groupby(
            [0, 1],  # (x1, y1)
            as_index=False).count())
    plt.suptitle("Feauture Overlap Counts")

def convertToImage(images,
                    nameLabel="Unknown",
                    index=0,
                    isShow=False,
                    folderSaving="outputImage"):
    plt.clf()
    cax = sns.heatmap(
        images[index],
        # cmap='hot',
        cmap='jet',
        linewidth=0.1,
        linecolor='dimgrey',
        square=False,
        cbar=False)
    cax.axis('off')
    figure = cax.get_figure()    
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
    figure.savefig(folderSaving+'/%s_%d.png'%(nameLabel, index), dpi=200)
    if isShow:
        plt.show()

class NonImageToImage:
    def __init__(self, file):
        print("[+] Read file csv %s"%(file))
        
        self.csv = pd.read_csv(file, engine="c")


        print("[+] filter rows with value NaN and Infinity")
        self.csv = self.csv.fillna(value=0)

        for i in range(len(self.csv['Flow Bytes/s'])):
            if self.csv['Flow Bytes/s'][i]=='Infinity':
                self.csv['Flow Bytes/s'][i]= '1040000001'

        for i in range(len(self.csv[' Flow Packets/s'])):
            if self.csv[' Flow Packets/s'][i]=='Infinity':
                self.csv[' Flow Packets/s'][i]= '2000001'

        self.csv.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.rowsNan = list(self.csv[self.csv.isna().any(axis=1)].index)
        self.csv.dropna(inplace=True)

        keysAttack = ["Heartbleed", "Web Attack Sql Injection", "Infiltration", "Web Attack XSS" "Web Attack Brute Force"
            "Bot", "DoS Slowhttptest", "DoS slowloris", "SSH-Patator", "FTP-Patator", "DoS GoldenEye", "DDoS"
            "PortScan", "DoS Hulk", "BENIGN"]
        self.dfs = []
        for k in keysAttack:
            dfTmp=self.csv[self.csv[' Label']==k].drop([' Label'],axis=1)
            if len(dfTmp) != 0:
                self.dfs.append({"index": dfTmp.index, "key": k})


    def getTrain_all_features(self):
        return self.train_all_features

    def getValid_all_features(self):
        return self.valid_all_features
    
    def getTest_all_features(self):
        return self.test_all_features

    def getCsv(self):
        return self.csv
    
    def getTrain_index(self):
        return self.train_index

    def getVal_index(self):
        return self.val_index

    def getClassCounts(self):
        class_counts = self.csv[self.train_classes].sum().to_frame(
            name="count").reset_index().rename(columns={"index": "class"})
        class_counts = class_counts.sort_values(by="count",
                                                ascending=False).reset_index(drop=True)
        return class_counts

    def convert2Image(self, isShow=False, folderSaving="outputImage"):
        if not os.path.exists(folderSaving):
                os.makedirs(folderSaving)

        featuresNotConvert = ['Flow ID', 'Source IP', 'Source Port', 'Destination IP', 'Destination Port', 'Protocol', 'Timestamp', 'Label']
        print("[+] get name collected column")
        numeric_features = [c for c in self.csv.columns if c.strip() not in featuresNotConvert]
        print("[+] get name unselected columns")
        expert_features  = [c for c in self.csv.columns if c.strip() in featuresNotConvert]

        kfolds = 10
        skf = MultilabelStratifiedKFold(n_splits=kfolds,
                                        shuffle=True,
                                        random_state=20)
        print("[+] get selected columns data")
        tmpCsv = self.csv
        for f in expert_features:
            tmpCsv = tmpCsv.drop(f, axis=1)
        self.train_classes = [c for c in tmpCsv.columns]

        label_counts = np.sum(tmpCsv, axis=0)
        y_labels = label_counts.index.tolist()

        self.train_index, self.val_index = list(skf.split(self.csv, self.csv[y_labels]))[0]
        self.train_index = np.setdiff1d(self.train_index, np.array(self.rowsNan))
        self.val_index = np.setdiff1d(self.val_index, np.array(self.rowsNan))

        print("[+] train data")
        self.train_all_features = self.csv.loc[self.train_index, numeric_features].copy().reset_index(drop=True).values
        self.valid_all_features = self.csv.loc[self.val_index, numeric_features].copy().reset_index(drop=True).values

        self.test_all_features = self.csv[numeric_features].copy().reset_index(drop=True).values

        print("[+] transform data train")
        all_scaler = image_transformer.LogScaler()
        self.train_all_features = all_scaler.fit_transform(self.train_all_features)
        self.valid_all_features = all_scaler.transform(self.valid_all_features)
        self.test_all_features = all_scaler.transform(self.test_all_features)
        # train_all_tsne = tsne_transform(train_all_features, perplexity=5)

        distance_metric = 'cosine'
        reducer = TSNE(
            n_components=2,
            metric=distance_metric,
            init='random',
            learning_rate='auto',
            n_jobs=-1
        )
        

        pixel_size = (227,227)
        all_it = image_transformer.ImageTransformer(
            feature_extractor=reducer, 
            pixels=pixel_size)

        resolution = 50
        all_it.fit(self.train_all_features, plot=False)
        plot_feature_density(
            all_it,
            pixels=resolution,
            title=
            f"All Feature Density Matrix of Training Set (Resolution: {resolution}x{resolution})",
            folderSaving=folderSaving
        )
        # divide to handle
        print("[+] divide to handle")
        arr = []
        index = (len(self.train_all_features) // 10000) + 1
        for i in range(0, index):
            arr.append(i*10000)
        arr.append(len(self.train_all_features))
        print("[+] Have %d part"%(len(arr)-1))
        
        for i in range(0, len(arr) - 1):
            print("[+] Tranform to image part %d"%(i))
            train_all_images = all_it.transform(self.train_all_features[arr[i]:arr[i+1]], empty_value=0, format="scalar")

            for distDf in self.dfs:
                indexs = distDf['index']
                label = distDf['key']

                for j in indexs:
                    if j < arr[i] or j >= arr[i+1]:
                        continue

                    convertToImage(
                        train_all_images,
                        nameLabel=label,
                        isShow=isShow,
                        index=j,
                        folderSaving=folderSaving
                    )
            del train_all_images