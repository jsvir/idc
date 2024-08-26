from torchvision.datasets import MNIST
from torch.utils.data import Dataset
import numpy as np
import torch
from sklearn import preprocessing
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import zscore
import matplotlib.pyplot as plt
from sklearn import datasets


class ClusteringDataset(Dataset):
    def __init__(self, data, labels=None, num_clusters=None):
        super().__init__()
        self.data = data
        self.labels = labels
        self._num_clusters = num_clusters
        if num_clusters is None and labels is None:
            raise ValueError("At least one of the values should be provided (labels/num_clusters)")
        self.print_stats()

    def __getitem__(self, index: int):
        if self.labels is None:
            return torch.tensor(self.data[index]).float()
        return torch.tensor(self.data[index]).float(), torch.tensor(self.labels[index]).long()

    def __len__(self) -> int:
        return len(self.data)

    @property
    def num_clusters(self):
        return self._num_clusters if self._num_clusters is not None else len(np.unique(self.labels))

    def num_features(self):
        return self.data.shape[-1]

    def print_stats(self):
        print('X.shape: ', self.data.shape)
        print(f"X.min={self.data.min()}, X.max={self.data.max()}")
        if self.labels is not None:
            print('Y.shape: ', self.labels.shape)
            for y_u in np.unique(self.labels):
                print(f'{y_u}: {np.sum(self.labels == y_u)}')
            print(f"Y.min={self.labels.min()}, Y.max={self.labels.max()}")

    @classmethod
    def setup(cls, cfg):
        pass


class PBMC(ClusteringDataset):
    def __init__(self, data, targets):
        super().__init__(data, targets)

    @classmethod
    def setup(cls, cfg):
        data_dir = cfg.data_dir
        with np.load(f"{data_dir}/pbmc_x.npz") as data:
            X = data['arr_0']
        with np.load(f"{data_dir}/pbmc_y.npz") as data:
            Y = data['arr_0']
        Y = Y - Y.min()
        scaler = getattr(preprocessing, cfg.scaler)()
        X = scaler.fit_transform(X)
        return cls(X, Y)


class BIASE(ClusteringDataset):
    def __init__(self, data, targets):
        super().__init__(data, targets)

    @classmethod
    def setup(cls, cfg):
        name = 'biase'
        data_dir = cfg.data_dir
        dataset_x = f"{data_dir}/{name}/{name}_data.csv"
        dataset_y = f"{data_dir}/{name}/{name}_celldata.csv"
        with open(dataset_x) as r:
            data = [l.strip() for l in r.readlines()]
        cell_keys = data[0].split(',')[1:]
        rows = [np.array([float(v) for v in row.split(',')[1:]]).reshape((1, -1)) for row in data[1:]]
        X = BIASE.remove_zero_columns(
            np.concatenate(rows, axis=0).transpose())  # np.concatenate(rows, axis=0).transpose()
        with open(dataset_y) as r:
            y_data = [l.strip().split(',') for l in r.readlines()[1:]]
        cell2class = {row[0]: row[2] for row in y_data}
        class2count = {}
        for cell, clas in cell2class.items():
            class2count.setdefault(clas, 0)
            class2count[clas] += 1

        print(class2count)
        class2id = {c: i for i, c in enumerate(set(sorted(list(cell2class.values()))))}

        Y = []
        for cell_key in cell_keys:
            Y.append(class2id[cell2class[cell_key]])
        Y = np.array(Y).reshape(-1)
        X = BIASE.transform(X)

        X = np.log(1 + X)
        X = X + .001 * np.random.normal(0, 1, (X.shape))
        scaler = getattr(preprocessing, cfg.scaler)()
        X = scaler.fit_transform(X)
        return cls(X, Y)


class INTESTINE(ClusteringDataset):
    def __init__(self, data, targets):
        super().__init__(data, targets)

    @classmethod
    def setup(cls, cfg):
        scaler = getattr(preprocessing, cfg.scaler)()
        name = 'intestine'
        data_dir = cfg.data_dir
        dataset_x = f"{data_dir}/{name}/{name}_data.csv"
        dataset_y = f"{data_dir}/{name}/{name}_celldata.csv"
        with open(dataset_x) as r:
            data = [l.strip() for l in r.readlines()]
        cell_keys = data[0].split(',')[1:]
        rows = [np.array([float(v) for v in row.split(',')[1:]]).reshape((1, -1)) for row in data[1:]]
        X = np.concatenate(rows, axis=0).T
        with open(dataset_y) as r:
            y_data = [l.strip().split(',') for l in r.readlines()[1:]]
        cell2class = {row[0]: row[2] for row in y_data}
        class2count = {}
        for cell, clas in cell2class.items():
            class2count.setdefault(clas, 0)
            class2count[clas] += 1
        print(class2count)
        class2id = {c: i for i, c in enumerate(sorted(set(list(cell2class.values()))))}
        Y = []
        for cell_key in cell_keys:
            Y.append(class2id[cell2class[cell_key]])
        Y = np.array(Y).reshape(-1)
        X = scaler.fit_transform(X)
        return cls(X, Y)


class CNAE9(ClusteringDataset):
    def __init__(self, data, targets):
        super().__init__(data, targets)

    @classmethod
    def setup(cls, cfg):
        scaler = getattr(preprocessing, cfg.scaler)()
        data = np.loadtxt(f"{cfg.data_dir}/cnae_9_numpy.txt")
        X = data[:, :-1]
        Y = data[:, -1]
        Y = Y - Y.min()
        X = scaler.fit_transform(X)
        return cls(X, Y)


class MFEATZERNIKE(ClusteringDataset):
    def __init__(self, data, targets):
        super().__init__(data, targets)

    @classmethod
    def setup(cls, cfg):
        scaler = getattr(preprocessing, cfg.scaler)()
        data = np.loadtxt(f"{cfg.data_dir}/mfeat_zernike_numpy.txt")
        X = data[:, :-1]
        Y = data[:, -1]
        Y = Y - Y.min()
        X = scaler.fit_transform(X)
        return cls(X, Y)


class ALLAML(ClusteringDataset):
    def __init__(self, data, targets):
        super().__init__(data, targets)

    @classmethod
    def setup(cls, cfg):
        dataset = loadmat(f"{cfg.data_dir}/ALLAML.mat")
        X = dataset.get('X')
        Y = dataset.get('Y').reshape(-1)
        Y = Y - Y.min()
        scaler = getattr(preprocessing, cfg.scaler)()
        X = scaler.fit_transform(X)
        return cls(X, Y)


class PROSTATE(ClusteringDataset):
    def __init__(self, data, targets):
        super().__init__(data, targets)

    @classmethod
    def setup(cls, cfg):
        dataset = loadmat(f"{cfg.data_dir}/PROSTATE.mat")
        X = dataset.get('X')
        Y = dataset.get('Y').reshape(-1)
        Y = Y - Y.min()  # to start from zero
        scaler = getattr(preprocessing, cfg.scaler)()
        X = scaler.fit_transform(X)
        return cls(X, Y)


class TOX171(ClusteringDataset):
    def __init__(self, data, targets):
        super().__init__(data, targets)

    @classmethod
    def setup(cls, cfg):
        dataset = loadmat(f"{cfg.data_dir}/TOX171.mat")
        X = dataset.get('X')
        Y = dataset.get('Y').reshape(-1)
        Y = Y - Y.min()  # to start from zero
        scaler = getattr(preprocessing, cfg.scaler)()
        X = scaler.fit_transform(X)
        return cls(X, Y)


class SRBCT(ClusteringDataset):
    def __init__(self, data, targets):
        super().__init__(data, targets)

    @classmethod
    def setup(cls, cfg):
        dataset = loadmat(f"{cfg.data_dir}/SRBCT.mat")
        X = dataset.get('X')
        Y = dataset.get('Y').reshape(-1)
        Y = Y - Y.min()  # to start from zero
        scaler = getattr(preprocessing, cfg.scaler)()
        X = scaler.fit_transform(X)
        return cls(X, Y)


class MNIST60K(ClusteringDataset):
    def __init__(self, data, targets):
        super().__init__(data, targets)

    @classmethod
    def setup(cls, cfg):
        scaler = getattr(preprocessing, cfg.scaler)()
        X = MNIST(cfg.data_dir, train=True, download=True).data.reshape(-1, 784).cpu().numpy()
        Y = MNIST(cfg.data_dir, train=True, download=True).targets.cpu().numpy()
        X = scaler.fit_transform(X)
        return cls(X, Y)


class MNIST10K(ClusteringDataset):
    def __init__(self, data, targets):
        super().__init__(data, targets)

    @classmethod
    def setup(cls, cfg):
        scaler = getattr(preprocessing, cfg.scaler)()
        X = MNIST(cfg.data_dir, train=True, download=True).data.reshape(-1, 784).cpu().numpy()
        Y = MNIST(cfg.data_dir, train=True, download=True).targets.cpu().numpy()
        X = scaler.fit_transform(X)
        X = X[:10000]
        Y = Y[:10000]
        return cls(X, Y)


class NumpyTableDataset(ClusteringDataset):
    def __init__(self, data, labels=None, num_clusters=None):
        super().__init__(data, labels, num_clusters)

    @classmethod
    def setup(cls, filepath_samples: str, filepath_labels: str = None, num_clusters: int = None):
        """
        :param filepath_samples: the path to the npz file, the format of the numpy array should be NxD
        (number of samples x number of features)
        :param filepath_labels: the path to the npz file, the format of the numpy array should be N
        (number of samples)
        :param num_clusters: the integer number of expected clusters
        """
        with np.load(filepath_samples) as data:
            X = data['arr_0']

        if filepath_labels is not None:
            with np.load(filepath_labels) as data:
                Y = data['arr_0']
            X = preprocessing.StandardScaler().fit_transform(X)
            Y = Y - Y.min()
        else:
            Y = None
        return cls(X, Y, num_clusters)


def remove_zero_columns(X):
    non_zero_columns = []
    for col in range(X.shape[1]):
        if np.min(X[:, col]) == 0 and np.max(X[:, col]) == 0:
            continue
        else:
            non_zero_columns.append(col)
    X = X[:, non_zero_columns]
    return X


class Synthetic(Dataset):
    def __init__(self, X, Y):
        super().__init__()
        self.data = X
        self.targets = Y

    def __getitem__(self, index: int):
        x = self.data[index]
        return torch.tensor(x).float(), torch.tensor(self.targets[index]).long()

    def __len__(self) -> int:
        return len(self.data)

    @classmethod
    def setup(cls, num_samples=5000, num_features=3, num_clusters=3, num_noise_dims=10):
        """
        Make num_clusters + 1 clusters in 3d and adds additional num_noise_dims noise features
        :param num_samples: number of samples in the dataset
        :param num_features: number of features in the dataset
        :param num_clusters: number of clusters in the dataset
        :param num_noise_dims: number of noise dimensions in addition to num_features
        :return: generates a dataset
        """
        x_2d, y_2d = datasets.make_blobs(num_samples, num_features-1, centers=num_clusters, cluster_std=.5,
                                         random_state=0)
        # split the points for cluster==2 into 2 clusters:
        max_x = x_2d[:, 1].max()
        min_x = x_2d[:, 1].min()
        x_y_2 = x_2d[y_2d == 2][:, 1]
        x_y_2 = MinMaxScaler((0, 1)).fit_transform(x_y_2.reshape(-1, 1)).reshape(-1)
        x_y_2 = MinMaxScaler((min_x, max_x)).fit_transform(x_y_2.reshape(-1, 1)).reshape(-1)
        x_2d[:, 1][y_2d == 2] = x_y_2

        z = np.random.rand(num_samples)
        y_2d[(y_2d == 2) & (z > 0.5)] = 3

        x_2d[:, 0][y_2d == 0] = x_2d[:, 0][y_2d == 1]

        bg = np.random.normal(loc=0, scale=0.01, size=(num_samples, num_noise_dims))
        X = np.concatenate([x_2d, z.reshape(-1, 1), bg], axis=1)
        X[:, 2][y_2d == 3] = X[:, 2][y_2d == 3] + 0.5  # separate in z axis
        X[:, 2][y_2d == 0] = MinMaxScaler(
            (X[:, 2][(y_2d == 3) | (y_2d == 2)].min(), X[:, 2][(y_2d == 3) | (y_2d == 2)].max())).fit_transform(
            X[:, 2][y_2d == 0].reshape(-1, 1)).reshape(-1)
        X[:, 2][y_2d == 1] = MinMaxScaler(
            (X[:, 2][(y_2d == 3) | (y_2d == 2)].min(), X[:, 2][(y_2d == 3) | (y_2d == 2)].max())).fit_transform(
            X[:, 2][y_2d == 1].reshape(-1, 1)).reshape(-1)

        Y = y_2d

        X4 = X[Y == 3]
        max_len = len(X4)
        X1 = X[Y == 0][:max_len, :]
        X2 = X[Y == 1][:max_len, :]
        X3 = X[Y == 2][:max_len, :]

        Y1 = Y[Y == 0][:max_len]
        Y2 = Y[Y == 1][:max_len]
        Y3 = Y[Y == 2][:max_len]
        Y4 = Y[Y == 3][:max_len]
        X = np.concatenate([X1, X2, X3, X4], axis=0)
        Y = np.concatenate([Y1, Y2, Y3, Y4], axis=0)

        print("Class stats:")
        for y_i in np.unique(Y):
            print(f"{y_i}: {len(Y[Y == y_i])} samples")
        X[:, :3] = zscore(X[:, :3])

        plt.style.use('classic')
        plt.rcParams['axes.spines.right'] = False
        plt.rcParams['axes.spines.top'] = False
        fig = plt.figure()
        fig.set_facecolor('w')
        plt.scatter(X[:, 0], X[:, 1], c=Y, s=100, alpha=0.8, cmap='viridis', edgecolor='k', linewidth=2)
        plt.xlabel('$X_1$', fontsize=30)
        plt.ylabel('$X_2$', fontsize=30)
        plt.tight_layout()
        plt.xticks([])
        plt.yticks([])
        plt.savefig("synth_X_1_X_2.png")

        plt.clf()
        fig = plt.figure()
        fig.set_facecolor('w')
        plt.scatter(X[:, 0], X[:, 2], c=Y, s=100, alpha=0.8, cmap='viridis', edgecolor='k', linewidth=2)
        plt.xlabel('$X_1$', fontsize=30)
        plt.ylabel('$X_3$', fontsize=30)
        plt.tight_layout()
        plt.xticks([])
        plt.yticks([])
        plt.savefig("synth_X_1_X_3.png")
        return cls(X, Y)

    def num_classes(self):
        return len(np.unique(self.targets))

    def num_features(self):
        return self.data.shape[-1]