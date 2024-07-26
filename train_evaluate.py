from itertools import chain
import torch
import math
from omegaconf import OmegaConf
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule
import numpy as np
from pytorch_lightning import Trainer, seed_everything
import os
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, silhouette_score, davies_bouldin_score
import argparse
import dataset
from model import clustering_head, aux_classifier_head, EncoderDecoder, GatingNet


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
        self.train_dataset = getattr(dataset, cfg.dataset).setup(cfg)
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

        self.val_cluster_list = []
        self.val_cluster_list_gated = []
        self.val_label_list = []
        self.open_gates = []
        self.val_embs_list = []

        self.best_acc = - 100
        self.best_ari = - 100
        self.best_nmi = - 100
        self.best_local_feats = None
        self.best_global_feats = None
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

    def update_stats(self, acc, ari, nmi, local_feats, global_feats):
        if self.best_acc <= acc:
            self.best_acc = acc
            self.best_ari = ari
            self.best_nmi = nmi
            self.best_local_feats = local_feats
            self.best_global_feats = global_feats

    def global_gates_step(self, x):
        gates = self.gating_net.get_gates(x)
        ae_emb = self.encdec.encoder(x * gates)
        cluster_logits = self.clustering_head(ae_emb)
        y_hat = cluster_logits.argmax(dim=-1)
        glob_gates_mu, glob_gates = self.gating_net.global_forward(y_hat)
        reg_loss = self.gating_net.regularization(glob_gates_mu)
        aux_y_hat = self.aux_classifier_head(x * gates * glob_gates)
        aux_loss = F.cross_entropy(aux_y_hat, y_hat)
        self.log('train/glob_gates_reg_loss', reg_loss.item())
        self.log('train/glob_gates_ce_loss', aux_loss.item())
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

    def training_step(self, batch, batch_idx):
        ae_opt, clust_opt, glob_gates_opt = self.optimizers()
        pretrain_sched, sch = self.lr_schedulers()
        x, _ = batch
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

    def validation_step(self, batch, batch_idx):
        x, y = batch
        gates = self.gating_net.get_gates(x)
        ae_emb = self.encdec.encoder(x * gates)
        cluster_logits = self.clustering_head(ae_emb)
        y_hat = cluster_logits.argmax(dim=-1)
        self.val_cluster_list.append(y_hat.cpu())
        self.val_label_list.append(y.cpu())
        self.open_gates.append(self.gating_net.num_open_gates(x))
        self.val_embs_list.append(ae_emb)

    def on_validation_epoch_start(self):
        self.val_cluster_list = []
        self.val_cluster_list_gated = []
        self.val_label_list = []
        self.open_gates = []
        self.val_embs_list = []

    @staticmethod
    def cluster_match(cluster_mtx, label_mtx, n_classes=10, print_result=True):
        cluster_indx = list(cluster_mtx.unique())
        assigned_label_list = []
        assigned_count = []
        while (len(assigned_label_list) <= n_classes) and len(cluster_indx) > 0:
            max_label_list = []
            max_count_list = []
            for indx in cluster_indx:
                mask = cluster_mtx == indx
                label_elements, counts = label_mtx[mask].unique(return_counts=True)
                for assigned_label in assigned_label_list:
                    counts[label_elements == assigned_label] = 0
                max_count_list.append(counts.max())
                max_label_list.append(label_elements[counts.argmax()])

            max_label = torch.stack(max_label_list)
            max_count = torch.stack(max_count_list)
            assigned_label_list.append(max_label[max_count.argmax()])
            assigned_count.append(max_count.max())
            cluster_indx.pop(max_count.argmax().item())
        total_correct = torch.tensor(assigned_count).sum().item()
        total_sample = cluster_mtx.shape[0]
        acc = total_correct / total_sample
        if print_result:
            print('{}/{} ({}%) correct'.format(total_correct, total_sample, acc * 100))
        else:
            return total_correct, total_sample, acc

    def on_validation_epoch_end(self):
        """ Based on https://github.com/zengyi-li/NMCE-release/blob/main/NMCE/func.py"""
        if not (self.ae_train and self.current_epoch < self.cfg.ae_pretrain_epochs) and self.current_epoch > 0:
            if self.current_epoch < self.cfg.ae_pretrain_epochs - 1:
                return
            else:
                cluster_mtx = torch.cat(self.val_cluster_list, dim=0)
            label_mtx = torch.cat(self.val_label_list, dim=0)
            _, _, acc_single = self.cluster_match(
                cluster_mtx,
                label_mtx,
                n_classes=label_mtx.max() + 1,
                print_result=False)
            if self.best_accuracy < acc_single:
                print("New best accuracy:", acc_single)
                self.best_accuracy = acc_single
                if self.cfg.save_seed_checkpoints:
                    meta_dict = {"gating": self.gating_net.state_dict(), "clustering": self.clustering_net.state_dict()}
                    torch.save(meta_dict, f'sparse_model_best_{self.cfg.dataset}_seed_{self.cfg.seed}.pth')

            nmi = normalized_mutual_info_score(label_mtx.numpy(), cluster_mtx.numpy())
            ari = adjusted_rand_score(label_mtx.numpy(), cluster_mtx.numpy())
            format_str = ''  # '_kmeans' if self.current_epoch == 9 else ''
            self.log(f'val/acc_single{format_str}', acc_single)  # this is ACC
            self.log(f'val/NMI{format_str}', nmi)
            self.log(f'val/ARI{format_str}', ari)
            self.log("val/num_open_gates", np.mean(self.open_gates).item())
            self.log("val/num_open_global_gates", self.gating_net.open_global_gates())
            if self.cfg.save_seed_checkpoints:
                meta_dict = {"gating": self.gating_net.state_dict(), "clustering": self.clustering_net.state_dict()}
                torch.save(meta_dict, f'sparse_model_last_{self.cfg.dataset}_seed_{self.cfg.seed}.pth')

            self.update_stats(acc_single, ari, nmi, np.mean(self.open_gates).item(),
                              self.gating_net.open_global_gates())

            try:
                silhouette_score_embs = silhouette_score(torch.cat(self.val_embs_list, dim=0).cpu().numpy(),
                                                         cluster_mtx.numpy())
                self.log(f'val/silhouette_score_embs', silhouette_score_embs)
                self.max_silhouette_score.append(silhouette_score_embs)
            except:
                pass
            try:
                dbi_score = davies_bouldin_score(torch.cat(self.val_embs_list, dim=0).cpu().numpy(),
                                                 cluster_mtx.numpy())
                self.log(f'val/dbi_score_embs', dbi_score)
                self.min_dbi_score.append(dbi_score)
            except:
                pass

    def mcrr_loss(self, c, logits):
        logprobs = torch.log_softmax(logits, dim=-1)
        prob = GumbleSoftmax(self.tau())(logprobs)
        _, compress_loss = self.mcrr(F.normalize(c), prob, num_classes=self.cfg.n_clusters)
        compress_loss /= c.size(1)
        self.log(f'train/compress_loss', compress_loss.item())
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
    if not cfg.validate:
        cfg.trainer.check_val_every_n_epoch = cfg.trainer.max_epochs + 1  # the validation will be never done

    with open(f"results_{os.path.basename(__file__)}.txt", mode='a') as f:
        header = '\t'.join(['seed', 'acc', 'ari', 'nmi', 'local_gates', 'global_gates',
                            'topk_max_silhouette_score', 'topk_min_dbi_score'])
        f.write(f"{header}\n")
        f.flush()
    for seed in range(cfg.seeds):
        cfg.seed = seed
        seed_everything(seed)
        np.random.seed(seed)
        if not os.path.exists(cfg.dataset):
            os.makedirs(cfg.dataset)
        model = BaseModule(cfg)
        logger = TensorBoardLogger(cfg.dataset, name=os.path.basename(__file__), log_graph=False)
        trainer = Trainer(**cfg.trainer, callbacks=[LearningRateMonitor(logging_interval='step')])
        trainer.logger = logger
        trainer.fit(model)
        topk_max_siluetter_score = np.mean(sorted(model.max_silhouette_score, reverse=True)[:10])
        topk_min_dbi_score = np.mean(sorted(model.max_silhouette_score)[:10])
        results_str = '\t'.join(
            [f'{seed}',
             f'{model.best_acc}',
             f'{model.best_ari}',
             f'{model.best_nmi}',
             f'{model.best_local_feats}',
             f'{model.best_global_feats}',
             f'{topk_max_siluetter_score}',
             f'{topk_min_dbi_score}',
             ])
        with open(f"results_{os.path.basename(__file__)}.txt", mode='a') as f:
            f.write(f"{results_str}\n")
            f.flush()
