# Copyright (c) 2018 Rui Shu
import torch
from codebase import utils as ut
from codebase.models import nns
from torch import nn
from torch.nn import functional as F

class VAE(nn.Module):
    def __init__(self, nn='v1', name='vae', z_dim=2):
        super().__init__()
        self.name = name
        self.z_dim = z_dim
        # Small note: unfortunate name clash with torch.nn
        # nn here refers to the specific architecture file found in
        # codebase/models/nns/*.py
        nn = getattr(nns, nn)
        self.enc = nn.Encoder(self.z_dim)
        self.dec = nn.Decoder(self.z_dim)


        # Set prior as fixed parameter attached to Module
        self.z_prior_m = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.z_prior_v = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.z_prior = (self.z_prior_m, self.z_prior_v)

    def negative_elbo_bound(self, x):
        """
        Computes the Evidence Lower Bound, KL and, Reconstruction costs

        Args:
            x: tensor: (batch, dim): Observations

        Returns:
            nelbo: tensor: (): Negative evidence lower bound
            kl: tensor: (): ELBO KL divergence to prior
            rec: tensor: (): ELBO Reconstruction term
        """
        ################################################################################
        # TODO: Modify/complete the code here
        # Compute negative Evidence Lower Bound and its KL and Rec decomposition
        #
        # Note that nelbo = kl + rec
        #
        # Outputs should all be scalar
        ################################################################################

        m, v = self.enc.encode(x)
        zs = ut.sample_gaussian(m, v)
        logits = self.dec.decode(zs)
        rec = torch.mean(ut.log_bernoulli_with_logits(x, logits))
        kl = torch.mean(ut.kl_normal(m, v, self.z_prior_m, self.z_prior_v))
        nelbo = -1 * (rec - kl)


#        print(nelbo, kl, rec)

        ################################################################################
        # End of code modification
        ################################################################################
        return nelbo, kl, rec

    def visualize(self):
        BATCH_SIZE = 200
        images = self.sample_x(BATCH_SIZE)
        images = images.view(-1, 28, 28)
        images = images.unsqueeze(1)
        images = images.repeat(1, 3, 1, 1)
        import torchvision
        grid = torchvision.utils.make_grid(images, 20)
        torchvision.utils.save_image(grid, 'vae-out.jpg')



    def negative_iwae_bound(self, x, iw):
        """
        Computes the Importance Weighted Autoencoder Bound
        Additionally, we also compute the ELBO KL and reconstruction terms

        Args:
            x: tensor: (batch, dim): Observations
            iw: int: (): Number of importance weighted samples

        Returns:
            niwae: tensor: (): Negative IWAE bound
            kl: tensor: (): ELBO KL divergence to prior
            rec: tensor: (): ELBO Reconstruction term
        """
        ################################################################################
        # TODO: Modify/complete the code here
        # Compute niwae (negative IWAE) with iw importance samples, and the KL
        # and Rec decomposition of the Evidence Lower Bound
        #
        # Outputs should all be scalar
        ################################################################################

        ## E_{z(1),...z(n)} {log p(x|z) + log p(z) - log q(z|x)}

        m, v = self.enc.encode(x)
        batch_m = m.unsqueeze(1)
        batch_m = batch_m.repeat(1, iw, 1) # dimension (batch, iw, 10)
        batch_v = v.unsqueeze(1)
        batch_v = batch_v.repeat(1, iw, 1)
        batch_x = x.unsqueeze(1)
        batch_x = batch_x.repeat(1, iw, 1)


        # log p(x|z)
        zs = ut.sample_gaussian(batch_m, batch_v)
        logits = self.dec.decode(zs)
        raw_probs = ut.log_bernoulli_with_logits(batch_x, logits)
        pxz = torch.mean(ut.log_mean_exp(raw_probs, dim=-1))


        # log p(z)
        batch_size = batch_m.shape[0]
        batch_z_prior_m = self.z_prior_m.view(1, 1, -1)
        batch_z_prior_m = batch_z_prior_m.repeat(batch_size, iw, 1)
        batch_z_prior_v = self.z_prior_v.view(1, 1, -1)
        batch_z_prior_v = batch_z_prior_v.repeat(batch_size, iw, 1)
        pz = ut.log_normal(zs, batch_z_prior_m, batch_z_prior_v) # (batch, iw)
        pz = torch.mean(ut.log_mean_exp(pz, dim=-1))


        # log q(z|x)
        qzx = ut.log_normal(zs, batch_m, batch_v)
        qzx = torch.mean(ut.log_mean_exp(qzx, dim=-1))


#        print(pxz, pz, qzx)
        niwae = -1 * (pxz + pz - qzx)

        rec = pxz - pz
        kl = pz - qzx
        ################################################################################
        # End of code modification
        ################################################################################
        return niwae, kl, rec

    def loss(self, x):
        nelbo, kl, rec = self.negative_elbo_bound(x)
        loss = nelbo

        summaries = dict((
            ('train/loss', nelbo),
            ('gen/elbo', -nelbo),
            ('gen/kl_z', kl),
            ('gen/rec', rec),
        ))

        return loss, summaries

    def sample_sigmoid(self, batch):
        z = self.sample_z(batch)
        return self.compute_sigmoid_given(z)

    def compute_sigmoid_given(self, z):
        logits = self.dec.decode(z)
        return torch.sigmoid(logits)

    def sample_z(self, batch):
        return ut.sample_gaussian(
            self.z_prior[0].expand(batch, self.z_dim),
            self.z_prior[1].expand(batch, self.z_dim))

    def sample_x(self, batch):
        z = self.sample_z(batch)
        return self.sample_x_given(z)

    def sample_x_given(self, z):
        return torch.bernoulli(self.compute_sigmoid_given(z))
