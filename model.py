import torch
import math


def init_weights_normal(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.normal_(m.weight, std=0.001)
        if 'bias' in vars(m).keys():
            m.bias.data.fill_(0.0)


def clustering_head(cfg):
    return torch.nn.Sequential(
        torch.nn.Linear(cfg.clustering_head[0], cfg.clustering_head[1]),
        torch.nn.BatchNorm1d(cfg.clustering_head[1]),
        torch.nn.ReLU(),
        torch.nn.Linear(cfg.clustering_head[1], cfg.n_clusters)).apply(init_weights_normal)


def aux_classifier_head(cfg):
    return torch.nn.Sequential(
        torch.nn.Linear(cfg.input_dim, cfg.aux_classifier[0]),
        torch.nn.BatchNorm1d(cfg.aux_classifier[0]),
        torch.nn.ReLU(),
        torch.nn.Linear(cfg.aux_classifier[0], cfg.n_clusters)).apply(init_weights_normal)


class EncoderDecoder(torch.nn.Module):
    def __init__(self, cfg):
        super(EncoderDecoder, self).__init__()
        self.cfg = cfg
        self.encoder = []
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        self.encoder.apply(init_weights_normal)
        self.decoder.apply(init_weights_normal)

    def build_encoder(self):
        layers = [
            torch.nn.Linear(self.cfg.input_dim, self.cfg.encdec[0]),
            torch.nn.BatchNorm1d(self.cfg.encdec[0]),
            torch.nn.ReLU()
        ]
        hidden_layers = len(self.cfg.encdec) // 2 + 1
        for layer_idx in range(1, hidden_layers):
            if layer_idx == hidden_layers - 1:
                layers += [torch.nn.Linear(self.cfg.encdec[layer_idx - 1], self.cfg.encdec[layer_idx])]
            else:
                layers += [
                    torch.nn.Linear(self.cfg.encdec[layer_idx - 1], self.cfg.encdec[layer_idx]),
                    torch.nn.BatchNorm1d(self.cfg.encdec[layer_idx]),
                    torch.nn.ReLU()
                ]
        return torch.nn.Sequential(*layers)

    def build_decoder(self):
        hidden_layers = len(self.cfg.encdec) // 2 + 1
        layers = []
        for layer_idx in range(hidden_layers, len(self.cfg.encdec)):
            layers += [
                torch.nn.Linear(self.cfg.encdec[layer_idx - 1], self.cfg.encdec[layer_idx]),
                torch.nn.BatchNorm1d(self.cfg.encdec[layer_idx]),
                torch.nn.ReLU()
            ]
        layers += [torch.nn.Linear(self.cfg.encdec[-1], self.cfg.input_dim)]
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.decoder(self.encoder(x))

class GatingNet(torch.nn.Module):
    def __init__(self, cfg):
        super(GatingNet, self).__init__()
        self.cfg = cfg
        self._sqrt_2 = math.sqrt(2)
        self.sigma = 0.5
        self.local_gates = torch.nn.Sequential(
            torch.nn.Linear(cfg.input_dim, cfg.gates_hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(cfg.gates_hidden_dim, cfg.input_dim),
            torch.nn.Tanh()
        )
        self.local_gates.apply(self.init_weights)
        self.global_gates_net = torch.nn.Embedding(self.cfg.n_clusters, self.cfg.input_dim)
        torch.nn.init.normal_(self.global_gates_net.weight, std=0.01)

    @staticmethod
    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.normal_(m.weight, std=0.001)
            if 'bias' in vars(m).keys():
                m.bias.data.fill_(0.0)

    def global_forward(self, y):
        noise = torch.normal(mean=0, std=self.sigma, size=(y.size(0), self.cfg.input_dim),
                             device=self.global_gates_net.weight.device)
        z = torch.tanh(self.global_gates_net(y)) + .5 * noise * self.training
        gates = self.hard_sigmoid(z)
        return torch.tanh(self.global_gates_net(y)), gates

    def open_global_gates(self):
        return self.hard_sigmoid(torch.tanh(self.global_gates_net.weight)).sum(dim=1).mean().cpu().item()

    def forward(self, x):
        noise = torch.normal(mean=0, std=self.sigma, size=x.size(), device=x.device)
        mu = self.local_gates(x)
        z = mu + .5 * noise * self.training
        gates = self.hard_sigmoid(z)
        sparse_x = x * gates
        return mu, sparse_x, gates

    @staticmethod
    def hard_sigmoid(x):
        return torch.clamp(x + .5, 0.0, 1.0)

    def regularization(self, mu, reduction_func=torch.mean):
        return reduction_func(0.5 - 0.5 * torch.erf((-1 / 2 - mu) / self._sqrt_2))

    def get_gates(self, x):
        with torch.no_grad():
            gates = self.hard_sigmoid(self.local_gates(x))
        return gates

    def num_open_gates(self, x, ):
        return self.get_gates(x).sum(dim=1).cpu().median(dim=0)[0].item()