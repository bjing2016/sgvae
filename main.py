import argparse
import datetime
import time
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from models import GraphConstructor
from models import GraphDestructor


class SGVAE(nn.Module):
    def __init__(self, z_dim, dims):
        super(SGVAE, self).__init__()
        self.encoder = GraphDestructor(z_dim=z_dim, dims=dims) 
        self.decoder = GraphConstructor(z_dim=z_dim, dims=dims)
        self.z_prior = (nn.Parameter(torch.zeros([z_dim]), requires_grad=False),
                        nn.Parameter(torch.ones([z_dim]), requires_grad=False))

    def loss(self, x):
        # note that nothing is batched !
        z, pi, log_qzpi = encoder.encode(x)
        # z     := vector of dimension z_dim
        # pi    := vector of [[idx] + [order]]
        # log_q := scalar
        log_pz = log_gaussian(z, self.z_prior)
        # log_pz := scalar
        genGraph, log_px = decoder.decode(actions=pi)
        # genGraph := Graph
        # log_px := scalar
        unldr = (log_px + log_pz - log_qzpi).detach() # unnormalized log-density ratio ?
        loss = unldr * log_qzpi + Lambda * log_px
        return loss

def __main__():
    model = SGVAE()










    # z -> p_theta(x, pi)
