from itertools import chain
import torch
import math
from omegaconf import OmegaConf
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule
import matplotlib.pyplot as plt
import numpy as np
from pytorch_lightning import Trainer, seed_everything
import os
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from sklearn.metrics import silhouette_score, davies_bouldin_score
import argparse
from dataset import NumpyTableDataset
from model import clustering_head, aux_classifier_head, EncoderDecoder, GatingNet
import umap


class TotalCodingRateWithProjection(torch.nn.Module):
    """ Based on https://github.com/zengyi-li/NMCE-release/blob/main/NMCE/loss.py """

    def __init__(self, cfg):
        super().__init__()
        self.eps = cfg.gtcr_eps
        if cfg.gtcr_projection_dim is not None:
            self.random_matrix = torch.tensor(np.random.normal(
                loc=0.0,
                scale=1.0 / np.sqrt(cfg.gtcr_projection_dim),
                size=(cfg.input_dim, cfg.gtcr_projection_dim)
            )).float()
        else:
            self.random_matrix = None

    def compute_discrimn_loss(self, W):
        p, m = W.shape  # [d, B]
        I = torch.eye(p, device=W.device)
        scalar = p / (m * self.eps)
        logdet = torch.logdet(I + scalar * W.matmul(W.T))
        return logdet / 2.

    def forward(self, x):
        if self.random_matrix is not None:
            x = x @ self.random_matrix.to(x.device)
        return - self.compute_discrimn_loss(x.T)


class MaximalCodingRateReduction(torch.nn.Module):
    """ Based on https://github.com/zengyi-li/NMCE-release/blob/main/NMCE/loss.py """

    def __init__(self, eps=0.01, gamma=1, compress_only=False):
        super(MaximalCodingRateReduction, self).__init__()
        self.eps = eps
        self.gamma = gamma
        self.compress_only = compress_only

    def compute_discrimn_loss(self, W):
        p, m = W.shape
        I = torch.eye(p, device=W.device)
        scalar = p / (m * self.eps)
        logdet = torch.logdet(I + scalar * W.matmul(W.T))
        return logdet / 2.

    def compute_compress_loss(self, W, Pi):
        p, m = W.shape
        k, _, _ = Pi.shape
        I = torch.eye(p, device=W.device).expand((k, p, p))
        trPi = Pi.sum(2) + 1e-8
        scale = (p / (trPi * self.eps)).view(k, 1, 1)
        W = W.view((1, p, m))
        log_det = torch.logdet(I + scale * W.mul(Pi).matmul(W.transpose(1, 2)))
        compress_loss = (trPi.squeeze() * log_det / (2 * m)).sum()
        return compress_loss

    def forward(self, X, Y, num_classes=None):
        # This function support Y as label integer or membership probablity.
        if len(Y.shape) == 1:
            # if Y is a label vector
            if num_classes is None:
                num_classes = Y.max() + 1
            Pi = torch.zeros((num_classes, 1, Y.shape[0]), device=Y.device)
            for indx, label in enumerate(Y):
                Pi[label, 0, indx] = 1
        else:
            # if Y is a probility matrix
            if num_classes is None:
                num_classes = Y.shape[1]
            Pi = Y.T.reshape((num_classes, 1, -1))

        W = X.T
        compress_loss = self.compute_compress_loss(W, Pi)
        if not self.compress_only:
            discrimn_loss = self.compute_discrimn_loss(W)
            return discrimn_loss, compress_loss
        else:
            return None, compress_loss


class BaseModule(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.train_dataset = NumpyTableDataset.setup(
            filepath_samples=cfg.get("filepath_samples"),
            num_clusters=cfg.get("num_clusters", None)
        )
        self.val_dataset = self.train_dataset

        print(f"Dataset length: {self.train_dataset.__len__()}")
        self.cfg.input_dim = self.train_dataset.num_features()
        self.cfg.n_clusters = self.train_dataset.num_clusters
        self.batch_size = min(self.train_dataset.__len__(), cfg.batch_size)

        self.save_hyperparameters()
        self.best_evaluation_stats = {}
        self.ae_train = False
        self.automatic_optimization = False
        self.best_accuracy = - np.infty
        self.gating_net = GatingNet(self.cfg)
        self.encdec = EncoderDecoder(self.cfg)
        self.clustering_head = clustering_head(self.cfg)
        self.aux_classifier_head = aux_classifier_head(self.cfg)
        self.mcrr = MaximalCodingRateReduction(eps=self.cfg.eps, compress_only=True)
        self.gtcr_loss = TotalCodingRateWithProjection(self.cfg)

        self.open_gates = []
        self.val_embs_list = []

        self.max_silhouette_score = []
        self.min_dbi_score = []

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          drop_last=True,
                          shuffle=True,
                          num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          drop_last=False,
                          shuffle=False,
                          num_workers=0)

    def global_gates_step(self, x):
        gates = self.gating_net.get_gates(x)
        ae_emb = self.encdec.encoder(x * gates)
        cluster_logits = self.clustering_head(ae_emb)
        y_hat = cluster_logits.argmax(dim=-1)
        glob_gates_mu, glob_gates = self.gating_net.global_forward(y_hat)
        reg_loss = self.gating_net.regularization(glob_gates_mu)
        aux_y_hat = self.aux_classifier_head(x * gates * glob_gates)
        aux_loss = F.cross_entropy(aux_y_hat, y_hat)
        self.log('glob_gates_reg_loss', reg_loss.item())
        self.log('glob_gates_ce_loss', aux_loss.item())
        return aux_loss + self.cfg.global_gates_lambda * reg_loss

    def ae_step(self, x):
        if self.current_epoch > self.cfg.ae_non_gated_epochs:
            mu, _, gates = self.gating_net(x)
            reg_loss = self.gating_net.regularization(mu)
            gtcr_loss = self.gtcr_loss(gates) / x.size(0)
            self.log("pretrain/gates_reg_loss", reg_loss.item())
            self.log("pretrain/gates_tcr_loss", gtcr_loss.item())
            loss = self.cosine_increase_lambda(
                min_val=0.,
                max_val=self.cfg.local_gates_lambda
            ) * reg_loss + gtcr_loss * self.cfg.gtcr_lambda
        else:
            gates = torch.ones_like(x, device=x.device).float()
            loss = 0

        # task 1: reconstruct x from x
        x_recon = self.encdec(x)
        x_recon_loss = F.mse_loss(x_recon, x)
        self.log("pretrain/x_recon_loss", x_recon_loss.item())

        # task 2: reconstruct x from gated x:
        x_recon_from_gated = self.encdec(x * gates)
        x_from_gated_x_recon_loss = F.mse_loss(x_recon_from_gated, x)
        self.log("pretrain/x_from_gated_x_recon_loss", x_from_gated_x_recon_loss.item())

        # task 3: reconstruct x from randomly masked x
        mask_rnd = torch.rand(x.size()).to(x.device)
        mask = torch.ones(x.size()).to(x.device).float()
        mask[mask_rnd < self.cfg.mask_percentage] = 0
        x_recon_masked = self.encdec(x * mask)
        input_noised_recon_loss = F.mse_loss(x_recon_masked, x)
        self.log("pretrain/input_noised_recon_loss", input_noised_recon_loss.item())

        # task 4: reconstruct x from noisy embedding
        e = self.encdec.encoder(x)
        e = e * torch.normal(mean=1., std=self.cfg.latent_noise_std, size=e.size(), device=e.device)
        recon_noised = self.encdec.decoder(e)
        noised_aug_loss = F.mse_loss(recon_noised, x)
        self.log("pretrain/latent_noised_recon_loss", noised_aug_loss.item())

        # combined loss:
        loss = loss + x_recon_loss + x_from_gated_x_recon_loss + input_noised_recon_loss + noised_aug_loss
        return loss

    def training_step(self, x, batch_idx):
        ae_opt, clust_opt, glob_gates_opt = self.optimizers()
        pretrain_sched, sch = self.lr_schedulers()
        x = x.reshape(x.size(0), -1)

        # reconstruction step + local gates training
        if self.current_epoch <= self.cfg.ae_pretrain_epochs:
            ae_opt.zero_grad()
            loss = self.ae_step(x)
            self.manual_backward(loss)
            ae_opt.step()
            pretrain_sched.step()
            return

        # clusters compression step
        clust_opt.zero_grad()
        gates = self.gating_net.get_gates(x)
        ae_emb = self.encdec.encoder(x * gates)
        cluster_logits = self.clustering_head(ae_emb)
        loss = self.mcrr_loss(ae_emb, cluster_logits)
        self.manual_backward(loss)
        clust_opt.step()

        # global gates training
        if self.current_epoch >= self.cfg.start_global_gates_training_on_epoch:
            glob_gates_opt.zero_grad()
            loss = self.global_gates_step(x)
            self.manual_backward(loss)
            glob_gates_opt.step()
        sch.step()

    def configure_optimizers(self):
        pretrain_optimizer = torch.optim.Adam(
            params=chain(
                self.encdec.parameters(),
                self.gating_net.local_gates.parameters(),
            ),
            lr=self.cfg.lr.pretrain)

        cluster_optimizer = torch.optim.Adam(
            params=chain(
                self.clustering_head.parameters(),
            ),
            lr=self.cfg.lr.clustering)

        glob_gates_opt = torch.optim.SGD(
            params=chain(
                self.aux_classifier_head.parameters(),
                self.gating_net.global_gates_net.parameters(),
            ),
            lr=self.cfg.lr.aux_classifier)

        steps = self.train_dataset.__len__() // self.batch_size * (
                self.cfg.trainer.max_epochs - self.cfg.ae_pretrain_epochs)
        pretrain_steps = self.train_dataset.__len__() // self.batch_size * self.cfg.ae_pretrain_epochs
        # pretrain_steps = self.dataset.__len__() // self.batch_size * self.cfg.trainer.max_epochs
        print(f"Cosine annealing LR scheduling is applied during {steps} steps")
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=cluster_optimizer,
            T_max=steps,
            eta_min=self.cfg.sched.clustering_min_lr)
        pretrain_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=pretrain_optimizer,
            T_max=pretrain_steps,
            eta_min=self.cfg.sched.pretrain_min_lr)
        return [pretrain_optimizer, cluster_optimizer, glob_gates_opt], [pretrain_sched, sched]

    def cosine_increase_lambda(self, min_val, max_val):
        epoch = self.current_epoch - self.cfg.ae_pretrain_epochs
        total_epochs = self.cfg.ae_pretrain_epochs - self.cfg.ae_non_gated_epochs
        return min_val + 0.5 * (max_val - min_val) * (1. + np.cos(epoch * math.pi / total_epochs))

    def validation_step(self, x, batch_idx):
        if not (self.ae_train and self.current_epoch < self.cfg.ae_pretrain_epochs) and self.current_epoch > 0:
            gates = self.gating_net.get_gates(x)
            ae_emb = self.encdec.encoder(x * gates)
            cluster_logits = self.clustering_head(ae_emb)
            y_hat = cluster_logits.argmax(dim=-1)
            self.val_cluster_list.append(y_hat.cpu())
            self.open_gates.append(self.gating_net.num_open_gates(x))
            self.val_embs_list.append(ae_emb)

    def on_validation_epoch_start(self):
        self.val_cluster_list = []
        self.open_gates = []
        self.val_embs_list = []

    @staticmethod
    def plot_clustering(val_embs_list, cluster_mtx, current_epoch, silhouette, dbi):
        reducer = umap.UMAP(n_neighbors=10, min_dist=0.1, n_components=2, random_state=0)
        embedding = reducer.fit_transform(torch.cat(val_embs_list, dim=0).cpu().numpy())
        plt.figure(figsize=(10, 7))
        plt.scatter(embedding[:, 0], embedding[:, 1], c=cluster_mtx.numpy(), s=50, edgecolor='k')
        plt.title(f'Clustering (UMAP). Epoch: {current_epoch}. Silhouette: {silhouette:0.3f}. DBI: {dbi:0.3f}')
        plt.savefig(f"umap_epoch_{current_epoch}.png")

    def on_validation_epoch_end(self):
        if not (self.ae_train and self.current_epoch < self.cfg.ae_pretrain_epochs) and self.current_epoch > 0:
            if self.current_epoch < self.cfg.ae_pretrain_epochs - 1:
                return
            else:
                cluster_mtx = torch.cat(self.val_cluster_list, dim=0)
            self.log("num_open_gates", np.mean(self.open_gates).item())
            self.log("num_open_global_gates", self.gating_net.open_global_gates())
            if self.cfg.save_seed_checkpoints:
                meta_dict = {"gating": self.gating_net.state_dict(), "clustering": self.clustering_net.state_dict()}
                torch.save(meta_dict, f'sparse_model_last_{self.cfg.dataset}_seed_{self.cfg.seed}.pth')
            try:
                silhouette_score_embs = silhouette_score(torch.cat(self.val_embs_list, dim=0).cpu().numpy(),
                                                         cluster_mtx.numpy())
                self.log(f'silhouette_score_embs', silhouette_score_embs)
                self.max_silhouette_score.append(silhouette_score_embs)
            except:
                silhouette_score_embs = -1
            try:
                dbi_score = davies_bouldin_score(torch.cat(self.val_embs_list, dim=0).cpu().numpy(),
                                                 cluster_mtx.numpy())
                self.log(f'dbi_score_embs', dbi_score)
                self.min_dbi_score.append(dbi_score)
            except:
                dbi_score = 0

            self.plot_clustering(self.val_embs_list, cluster_mtx, self.current_epoch, silhouette_score_embs, dbi_score)

    def mcrr_loss(self, c, logits):
        logprobs = torch.log_softmax(logits, dim=-1)
        prob = GumbleSoftmax(self.tau())(logprobs)
        _, compress_loss = self.mcrr(F.normalize(c), prob, num_classes=self.cfg.n_clusters)
        compress_loss /= c.size(1)
        self.log(f'compress_loss', compress_loss.item())
        return compress_loss

    def tau(self):
        return self.cfg.tau


class GumbleSoftmax(torch.nn.Module):
    def __init__(self, tau, straight_through=False):
        super().__init__()
        self.tau = tau
        self.straight_through = straight_through

    def forward(self, logps):
        gumble = torch.rand_like(logps).log().mul(-1).log().mul(-1)
        logits = logps + gumble
        out = (logits / self.tau).softmax(dim=1)
        if not self.straight_through:
            return out
        else:
            out_binary = (logits * 1e8).softmax(dim=1).detach()
            out_diff = (out_binary - out).detach()
            return out_diff + out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.cfg)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    for seed in range(cfg.seeds):
        cfg.seed = seed
        seed_everything(seed)
        np.random.seed(seed)
        model = BaseModule(cfg)
        logger = TensorBoardLogger("logs", name=os.path.basename(__file__), log_graph=False)
        trainer = Trainer(**cfg.trainer, callbacks=[LearningRateMonitor(logging_interval='step')])
        trainer.logger = logger
        trainer.fit(model)
