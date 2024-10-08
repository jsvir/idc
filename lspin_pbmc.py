from itertools import chain
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from pytorch_lightning import Trainer, seed_everything
import argparse
import torch
import math
import numpy as np
import os
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
import platform


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--gated", action="store_true")
    parser.add_argument("--dataset", type=str, default="PBMC")
    parser.add_argument("--data_dir", type=str, default="C:/data/fs/pbmc" if platform.system() == "Windows" else ".")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--repitions", type=int, default=5)
    parser.add_argument("--lr", type=int, default=1e-3)

    # gatenet config
    parser.add_argument("--sigma", type=float, default=0.5)
    parser.add_argument("--reg_beta", type=float, default=10)
    parser.add_argument("--target_sparsity", type=int, default=0.9)
    parser.add_argument("--gates_lr", type=int, default=2e-3)


    # trainer config
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--deterministic", type=bool, default=True)
    parser.add_argument("--logger", type=bool, default=True)
    parser.add_argument("--log_every_n_steps", type=int, default=10)
    parser.add_argument("--check_val_every_n_epoch", type=int, default=1)
    parser.add_argument("--enable_checkpointing", type=bool, default=False)

    args = parser.parse_args(args)
    return args


class PBMC(Dataset):
    def __init__(self, X, Y):
        super().__init__()
        self.data = X
        self.targets = Y

    def __getitem__(self, index: int):
        x = self.data[index]
        x = x.reshape(-1)
        return torch.tensor(x).float(), torch.tensor(self.targets[index]).long()

    def __len__(self) -> int:
        return len(self.data)

    @classmethod
    def setup(cls, data_dir, test_size=0.2):
        with np.load(f"{data_dir}/pbmc_x.npz") as data:
            X = data['arr_0']
        with np.load(f"{data_dir}/pbmc_y.npz") as data:
            Y = data['arr_0']

        Y = Y - Y.min()
        X = StandardScaler().fit_transform(X)
        print(f'Dataset PBMC stats:')
        print('X.shape: ', X.shape)
        print('Y.shape: ', Y.shape)
        print(f"X.min={X.min()}, X.max={X.max()}")
        print(f"Y.min={Y.min()}, Y.max={Y.max()}")

        for y_uniq in np.unique(Y):
            print(f"Label {y_uniq} has {len(Y[Y == y_uniq])} samples")

        np.random.seed(1948)
        random_index = np.random.permutation(len(X))
        test_size = int(len(X) * test_size)
        x_test = X[random_index][:test_size]
        y_test = Y[random_index][:test_size]

        x_train = X[random_index][test_size:]
        y_train = Y[random_index][test_size:]

        print(f"Split to train/test: train {len(x_train)} test {len(x_test)}")
        return cls(x_train, y_train), cls(x_test, y_test)

    def num_classes(self):
        return len(np.unique(self.targets))

    def num_features(self):
        return self.data.shape[-1]


class BaseModule(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()
        self.best_evaluation_stats = {}
        self.automatic_optimization = False
        self.best_accuracy = - np.infty
        self.classifier_net = Classifier(cfg)
        self.val_cluster_list = []
        self.val_label_list = []
        self.best_acc = - 100

        if cfg.gated:
            self.gating_net = GatingNet(cfg)
            self.val_cluster_list_gated = []
            self.open_gates = []
            self.best_local_feats = None

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        sch = self.lr_schedulers()
        x, y = batch
        x = x.reshape(x.size(0), -1)
        opt.zero_grad()

        if hasattr(self, 'gating_net'):
            mu, _, gates = self.gating_net(x)
            ae_emb = self.classifier_net.encoder(x * gates)

            reg_loss = self.gating_net.regularization(mu)
            self.log("train/reg_loss", reg_loss.item())
        else:
            ae_emb = self.classifier_net.encoder(x)
            reg_loss = 0

        cluster_logits = self.classifier_net.head(ae_emb)
        ce_loss = F.cross_entropy(cluster_logits, y)

        self.log("train/ce_loss", ce_loss.item())
        loss = ce_loss + self.cfg.reg_beta * reg_loss
        self.manual_backward(loss)
        opt.step()
        sch.step()

        if self.global_step % 100 == 0:
            if hasattr(self, 'gating_net'):
                print(f"Epoch {self.current_epoch} "
                      f"step {self.global_step} "
                      f"train/reg_loss {reg_loss.item()} "
                      f"train/ce_loss {ce_loss.item()}")
    def configure_optimizers(self):

        if hasattr(self, 'gating_net'):
            params =[ {  # classifier
                "params": chain(
                self.classifier_net.encoder.parameters(),
                self.classifier_net.head.parameters()),
                "lr": self.cfg.lr,

            },
            {  # gates
                "params": self.gating_net.net.parameters(),
                "lr": self.cfg.gates_lr,
            }]

        else:
            params = chain(
                self.classifier_net.encoder.parameters(),
                self.classifier_net.head.parameters(),
            )
        optimizer = torch.optim.SGD(
            params=params,
            lr=self.cfg.lr)

        steps = self.train_dataset.__len__() // self.batch_size * self.cfg.max_epochs
        print(f"Cosine annealing LR scheduling is applied during {steps} steps")
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=steps,
            eta_min=1e-4)
        return [optimizer], [sched]

    def validation_step(self, batch, batch_idx):
        x, y = batch
        if hasattr(self, 'gating_net'):
            gates = self.gating_net.get_gates(x)
            ae_emb = self.classifier_net.encoder(x * gates)
            self.open_gates.append(self.gating_net.num_open_gates(x))
        else:
            ae_emb = self.classifier_net.encoder(x)
        cluster_logits = self.classifier_net.head(ae_emb)
        y_hat = cluster_logits.argmax(dim=-1)
        self.val_cluster_list.append(y_hat.cpu())
        self.val_label_list.append(y.cpu())

    def on_validation_epoch_start(self):
        self.val_cluster_list = []
        self.val_cluster_list_gated = []
        self.val_label_list = []
        if hasattr(self, 'gating_net'):
            self.open_gates = []

    def on_validation_epoch_end(self):
        if self.current_epoch > 0:
            cluster_mtx = torch.cat(self.val_cluster_list, dim=0)
            label_mtx = torch.cat(self.val_label_list, dim=0)
            acc = torch.mean((cluster_mtx == label_mtx).float()).item()
            if self.best_accuracy < acc:
                self.best_accuracy = acc
                if hasattr(self, 'gating_net'):
                    meta_dict = {"gating": self.gating_net.state_dict(), "clustering": self.classifier_net.state_dict()}
                    torch.save(meta_dict, f'sparse_model_best_pbmc_beta_{self.cfg.reg_beta}_seed_{self.cfg.seed}.pth')
                    print(f"New best accuracy: {acc} open gates: {np.mean(self.open_gates).item()}")
                else:
                    meta_dict = {"clustering": self.classifier_net.state_dict()}
                    torch.save(meta_dict, f'sparse_model_nogates_best_pbmc_seed_{self.cfg.seed}.pth')
                    print(f"New best accuracy: {acc}")
            format_str = ''  # '_kmeans' if self.current_epoch == 9 else ''
            self.log(f'val/acc_single{format_str}', acc)  # this is ACC
            if hasattr(self, 'gating_net'):
                self.log("val/num_open_gates", np.mean(self.open_gates).item())
                meta_dict = {"gating": self.gating_net.state_dict(), "clustering": self.classifier_net.state_dict()}
                torch.save(meta_dict, f'sparse_model_last_pbmc_beta_{self.cfg.reg_beta}_seed_{self.cfg.seed}.pth')
                self.update_stats(acc, np.mean(self.open_gates).item())
            else:
                meta_dict = {"clustering": self.classifier_net.state_dict()}
                torch.save(meta_dict, f'sparse_model_nogates_last_pbmc_seed_{self.cfg.seed}.pth')
                self.update_stats(acc, None)

    def update_stats(self, acc, local_feats=None):
        if self.best_acc <= acc:
            self.best_acc = acc
            if local_feats is not None:
                self.best_local_feats = local_feats


class ClassificationModule(BaseModule):
    def __init__(self, cfg):
        self.train_dataset, self.test_dataset = PBMC.setup(cfg.data_dir)
        print(f"Train Dataset length: {self.train_dataset.__len__()}")
        print(f"Test Dataset length: {self.test_dataset.__len__()}")
        cfg.input_dim = self.train_dataset.num_features()
        cfg.n_clusters = self.train_dataset.num_classes()
        self.batch_size = min(self.train_dataset.__len__(), cfg.batch_size)
        super().__init__(cfg)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          drop_last=True,
                          shuffle=True,
                          num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          drop_last=False,
                          shuffle=False,
                          num_workers=0)


class Classifier(torch.nn.Module):
    def __init__(self, cfg):
        super(Classifier, self).__init__()
        self.cfg = cfg
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(cfg.input_dim, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
        )
        self.head = torch.nn.Sequential(
            torch.nn.Linear(512, 2048),
            torch.nn.BatchNorm1d(2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, cfg.n_clusters),
        )

        self.encoder.apply(self.init_weights_normal)
        self.head.apply(self.init_weights_normal)

    @staticmethod
    def init_weights_normal(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)
            if 'bias' in vars(m).keys():
                m.bias.data.fill_(0.0)

    def pretrain_forward(self, x):
        return self.decoder(self.encoder(x))


class GatingNet(torch.nn.Module):
    def __init__(self, cfg):
        super(GatingNet, self).__init__()
        self.cfg = cfg
        self._sqrt_2 = math.sqrt(2)
        self.sigma = cfg.sigma
        self.net = torch.nn.Sequential(
            torch.nn.Linear(cfg.input_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, cfg.input_dim),
            torch.nn.Tanh()
        )
        self.net.apply(self.init_weights)
        
    def init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)
            if m.out_features == self.cfg.input_dim:
                m.bias.data.fill_(.5)
            else:
                m.bias.data.fill_(0.0)

    def global_forward(self, batch_size, y):
        noise = torch.normal(mean=0, std=self.sigma, size=(batch_size, self.cfg.input_dim),
                             device=self.global_gates_net.weight.device)
        z = torch.tanh(self.global_gates_net(y)).reshape(1, -1).repeat(batch_size, 1) + noise * self.training
        gates = self.hard_sigmoid(z)
        return torch.tanh(self.global_gates_net(y)), gates

    def open_global_gates(self):
        return self.hard_sigmoid(torch.tanh(self.global_gates_net.weight)).sum(dim=1).mean().cpu().item()

    def forward(self, x):
        noise = torch.normal(mean=0, std=self.sigma, size=x.size(), device=x.device)
        mu = self.net(x)
        z = mu + noise * self.training
        gates = self.hard_sigmoid(z)
        sparse_x = x * gates
        return mu, sparse_x, gates

    @staticmethod
    def hard_sigmoid(x):
        return torch.clamp(x + .5, 0.0, 1.0)

    def regularization(self, mu, reduction_func=torch.mean):
        return max(reduction_func(0.5 - 0.5 * torch.erf((-0.5 - mu) / (0.5 * self._sqrt_2))),
                   torch.tensor(1 - self.cfg.target_sparsity, device=mu.device, dtype=mu.data.dtype))

    def get_gates(self, x):
        with torch.no_grad():
            gates = self.hard_sigmoid(self.net(x))
        return gates

    def num_open_gates(self, x):
        return torch.sum(self.get_gates(x) > 0).item() / x.size(0)


def train_test(cfg):
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    gated_str = "_gated" if cfg.gated else ""
    with open(f"results_{os.path.basename(__file__)}{gated_str}_reg_beta_{cfg.reg_beta}.txt", mode='w') as f:

        header = '\t'.join(['seed', 'acc', 'local_gates'])
        f.write(f"{header}\n")
        f.flush()

        for seed in range(cfg.repitions):
            cfg.seed = seed
            seed_everything(seed)
            np.random.seed(seed)
            if not os.path.exists(cfg.dataset):
                os.makedirs(cfg.dataset)
            model = ClassificationModule(cfg)
            logger = TensorBoardLogger(cfg.dataset, name=os.path.basename(__file__), log_graph=False)
            trainer = Trainer(
                devices=cfg.devices,
                accelerator=cfg.accelerator,
                max_epochs=cfg.max_epochs,
                deterministic=cfg.deterministic,
                logger=cfg.logger,
                log_every_n_steps=cfg.log_every_n_steps,
                check_val_every_n_epoch=cfg.check_val_every_n_epoch,
                enable_checkpointing=cfg.enable_checkpointing,
                callbacks=[LearningRateMonitor(logging_interval='step')]
            )
            trainer.logger = logger
            trainer.fit(model)
            if cfg.gated:
                results_str = '\t'.join([f'{seed}', f'{model.best_acc}', f'{model.best_local_feats}'])
            else:
                results_str = '\t'.join([f'{seed}', f'{model.best_acc}'])
            f.write(f"{results_str}\n")
            f.flush()


if __name__ == "__main__":
    cfg = parse_args(None)
    train_test(cfg)
