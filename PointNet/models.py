import torch
from torch import nn, norm, normal
import torch.utils.data
import torch.nn.functional as F

from encoders import  ExplicitEncoder#ExplicitEncoder, GaussianEncoder, RaisedCosEncoder, HalfCosEncoder, SincEncoder
from encoder import GaussianEncoder


class ExplicitBasisPointNet(nn.Module):
    def __init__(self, n_primitives,
                 in_dim=3,
                 cloud_range=torch.tensor([[-1, 1], [-1, 1], [-1, 1]]),
                 init_mode='random',
                 normalized=True,
                 global_dim=1024,
                 k=40,
                 use_tnet=True,
                 isotropic=False,
                 diagonal=False,
                 **kwargs):
        super().__init__()
        tnet_primitives = n_primitives
        self.feat = ExplicitEncoder(
            in_dim=in_dim,
            out_dim=global_dim,
            n_primitives=n_primitives,
            cloud_range=cloud_range,
            init_mode=init_mode,
            normalized=normalized
        )
        self.fc1 = nn.Linear(global_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
        self.use_tnet = use_tnet
        if use_tnet:
            self.tnet = ExplicitBasisTNet(n_primitives=tnet_primitives,
                                          in_dim=in_dim,
                                          cloud_range=cloud_range,
                                          normalized=normalized,
                                          init_mode=init_mode,
                                          isotropic=isotropic,
                                          diagonal=diagonal,
                                          )

    def forward(self, x):
        #UNCOMMENT THIS
        if self.use_tnet:
            trans = self.tnet(x)
            x = torch.bmm(x, trans)
        x = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x


class ExplicitBasisTNet(nn.Module):
    def __init__(self, n_primitives,
                 in_dim=3,
                 cloud_range=torch.tensor([[-1, 1], [-1, 1], [-1, 1]]),
                 init_mode='random',
                 normalized=True,
                 isotropic=False,
                 diagonal=False,
                 ):
        super().__init__()
        self.feat = ExplicitEncoder(
            in_dim=in_dim,
            out_dim=1024,
            n_primitives=n_primitives,
            cloud_range=cloud_range,
            init_mode=init_mode,
            normalized=normalized
        )
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        k = in_dim
        self.fc3 = nn.Linear(256, k**2)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
        self.k = k

    def forward(self, x):
        #UNCOMMENT THIS
        x = self.feat(x)
        # probably not necessary, check
        x = x.view(-1, 1024)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        iden = torch.eye(self.k, device=x.device, dtype=torch.float32, requires_grad=True).flatten().view(1, self.k * self.k).repeat(x.shape[0], 1)
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class GaussianBasisPointNet(ExplicitBasisPointNet):
    def __init__(self, n_primitives,
                 in_dim=3,
                 cloud_range=torch.tensor([[-1, 1], [-1, 1], [-1, 1]]),
                 init_mode='random',
                 normalized=True,
                 global_dim=1024,
                 k=40,
                 use_tnet=True,
                 isotropic=False,
                 diagonal=False,
                 **kwargs):
        super().__init__(
            n_primitives,
            in_dim,
            cloud_range,
            init_mode,
            normalized,
            global_dim,
            k,
            use_tnet,
            isotropic,
            **kwargs,
        )
        tnet_primitives = n_primitives
        self.feat = GaussianEncoder(
            in_dim=in_dim,
            out_dim=global_dim,
            n_primitives=n_primitives,
            cloud_range=cloud_range,
            init_mode=init_mode,
            normalized=normalized,
            isotropic=isotropic,
            diagonal=diagonal,
        )
        if use_tnet:
            self.tnet = GaussianBasisTNet(n_primitives=tnet_primitives,
                                          in_dim=in_dim,
                                          cloud_range=cloud_range,
                                          normalized=normalized,
                                          init_mode=init_mode,
                                          isotropic=isotropic,
                                          diagonal=diagonal,
                                          )


class GaussianBasisTNet(ExplicitBasisTNet):
    def __init__(self, n_primitives,
                 in_dim=3,
                 cloud_range=torch.tensor([[-1, 1], [-1, 1], [-1, 1]]),
                 init_mode='random',
                 normalized=True,
                 isotropic=False,
                 diagonal=False,
                 ):
        super().__init__(
            n_primitives,
            in_dim,
            cloud_range,
            init_mode,
            normalized,
            isotropic,
            diagonal,
        )
        self.feat = GaussianEncoder(
            in_dim=in_dim,
            out_dim=1024,
            n_primitives=n_primitives,
            cloud_range=cloud_range,
            init_mode=init_mode,
            normalized=normalized,
            isotropic=isotropic,
            diagonal=diagonal,
        )
