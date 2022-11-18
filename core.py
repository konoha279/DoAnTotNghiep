from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import image_transformer

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

def plot_feature_density(it, pixels=100, show_grid=True, title=None):
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
    plt.show()

    # Feature Overlapping Counts
    gene_overlap = (
        pd.DataFrame(all_it._coords.T).assign(count=1).groupby(
            [0, 1],  # (x1, y1)
            as_index=False).count())
    print(gene_overlap["count"].describe())
    print(gene_overlap["count"].hist())
    plt.suptitle("Feauture Overlap Counts")

def plot_feature_images(it, labels, images, classes, title=None, n_cols=2, top_k_classes=0):
    # Create subplots
    fig, ax = plt.subplots(top_k_classes // n_cols, n_cols, figsize=(12, 12))

    for i in range(0, top_k_classes // n_cols):
        for j in range(n_cols):
            class_rows = labels[labels[classes[i + j]] > 0]
            # Select the random row of each class
            sample_index = np.random.choice(class_rows.index.values, size=1)[0]
            cax = sns.heatmap(
                images[sample_index],
                # cmap='hot',
                cmap='jet',
                linewidth=0.01,
                linecolor='dimgrey',
                square=False,
                ax=ax[i, j],
                cbar=True)
            cax.axis('off')

            ax[i, j].set_title(f"{classes[i*n_cols + j]} (index: {sample_index})",
                               fontsize=14)

    plt.rcParams.update({'font.size': 14})
    if title is not None:
        plt.suptitle(title, fontsize=20)

    plt.tight_layout()
    fig.subplots_adjust(top=0.9)
    plt.show()

def plot_class_feature_images(it,
                              labels,
                              images,
                              target_class,
                              title=None,
                              n_rows=2,
                              n_cols=2,
                              index=0,
                              isShow=False,
                              folderSaving="outputImage"):
    for i in range(len(images)):
        cax = sns.heatmap(
            images[i],
            # cmap='hot',
            cmap='jet',
            linewidth=0.001,
            linecolor='dimgrey',
            square=False,
            cbar=True)
        figure = cax.get_figure()    
        figure.savefig(folderSaving+'/%d.png'%(index+i), dpi=400)
        if isShow:
            plt.show()
        plt.clf()

class LogScaler:
    """Log normalize and scale data

    Log normalization and scaling procedure as described as norm-2 in the
    DeepInsight paper supplementary information.
    
    Note: The dimensions of input matrix is (N samples, d features)
    """
    def __init__(self):
        self._min0 = None
        self._max = None

    """
    Use this as a preprocessing step in inference mode.
    """
    def fit(self, X, y=None):
        # Min. of training set per feature
        self._min0 = X.min(axis=0)

        # Log normalized X by log(X + _min0 + 1)
        X_norm = np.log(
            X +
            np.repeat(np.abs(self._min0)[np.newaxis, :], X.shape[0], axis=0) +
            1).clip(min=0, max=None)

        # Global max. of training set from X_norm
        self._max = X_norm.max()

    """
    For training set only.
    """
    def fit_transform(self, X, y=None):
        # Min. of training set per feature
        self._min0 = X.min(axis=0)

        # Log normalized X by log(X + _min0 + 1)
        X_norm = np.log(
            X +
            np.repeat(np.abs(self._min0)[np.newaxis, :], X.shape[0], axis=0) +
            1).clip(min=0, max=None)

        # Global max. of training set from X_norm
        self._max = X_norm.max()

        # Normalized again by global max. of training set
        return (X_norm / self._max).clip(0, 1)

    """
    For validation and test set only.
    """
    def transform(self, X, y=None):
        # Adjust min. of each feature of X by _min0
        for i in range(X.shape[1]):
            X[:, i] = X[:, i].clip(min=self._min0[i], max=None)

        # Log normalized X by log(X + _min0 + 1)
        X_norm = np.log(
            X +
            np.repeat(np.abs(self._min0)[np.newaxis, :], X.shape[0], axis=0) +
            1).clip(min=0, max=None)

        # Normalized again by global max. of training set
        return (X_norm / self._max).clip(0, 1)

class NonImageToImage:
    def __init__(self, file, columnsException):
        print("[+] Read file csv %s"%(file))
        self.csv = pd.read_csv(file, engine="c")

        print("[+] filter and remove rows with value NaN")
        self.csv.replace([np.inf, -np.inf], np.nan, inplace=True)
        rowsNan = list(self.csv[self.csv.isna().any(axis=1)].index)
        self.csv.dropna(inplace=True)

        print("[+] get name collected column")
        numeric_features = [c for c in self.csv.columns if c.strip() not in columnsException]
        print("[+] get name unselected columns")
        expert_features  = [c for c in self.csv.columns if c.strip() in columnsException]

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
        self.train_index = np.setdiff1d(self.train_index, np.array(rowsNan))
        self.val_index = np.setdiff1d(self.val_index, np.array(rowsNan))

        print("[+] train data")
        self.train_all_features = self.csv.loc[self.train_index, numeric_features].copy().reset_index(drop=True).values
        self.valid_all_features = self.csv.loc[self.val_index, numeric_features].copy().reset_index(drop=True).values

        self.test_all_features = self.csv[numeric_features].copy().reset_index(drop=True).values

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
        all_it.fit(self.train_all_features, plot=False)

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
            global top_k_classes
            top_k_classes = 2
            class_counts = self.csv[self.train_classes].sum().to_frame(
                name="count").reset_index().rename(columns={"index": "class"})
            class_counts = class_counts.sort_values(by="count",
                                                    ascending=False).reset_index(drop=True)
            sample_labels = self.csv.iloc[self.train_index, :].copy().reset_index(drop=True)
            sample_labels = sample_labels[
                class_counts["class"].values.tolist()]
            sample_labels = sample_labels[sample_labels.sum(axis=1) > 0]

            class_index = 3
            top_classes = class_counts["class"].values.tolist()
            plot_class_feature_images(
                all_it,
                sample_labels,
                train_all_images,
                target_class=top_classes[class_index],
                isShow=isShow,
                index=arr[i],
                folderSaving=folderSaving
            )
