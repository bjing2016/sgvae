import argparse
import datetime
import time
import torch
import dgl
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from models import GraphConstructor
from models import GraphDestructor
from chem_utils import MoleculeDataset
import torch.nn as nn


def main():
    dataset = MoleculeDataset('ChEMBL', 'canonical', ['train', 'val'])
    train_loader = DataLoader(dataset.train_set, batch_size=args['batch_size'],
                              shuffle=True, collate_fn=dataset.collate)
    val_loader = DataLoader(dataset.val_set, batch_size=args['batch_size'],
                            shuffle=True, collate_fn=dataset.collate)
    model = SGVAE(atom_types=dataset.atom_types,
                  bond_types=dataset.bond_types,
                  node_hidden_size=args['node_hidden_size'],
                  num_prop_rounds=args['num_propagation_rounds'],
                  dropout=args['dropout'])
    if args['num_processes'] == 1:
        from utils import Optimizer
        optimizer = Optimizer(args['lr'], Adam(model.parameters(), lr=args['lr']))
    else:
        from utils import MultiProcessOptimizer
        optimizer = MultiProcessOptimizer(args['num_processes'], args['lr'],
                                          Adam(model.parameters(), lr=args['lr']))

    if rank == 0:
        t2 = time.time()
    best_val_prob = 0

    # Training
    for epoch in range(args['nepochs']):
        model.train()
        if rank == 0:
            print('Training')

        for i, data in enumerate(train_loader):
            log_prob = model(actions=data, compute_log_prob=True)
            prob = log_prob.detach().exp()

            loss_averaged = - log_prob
            prob_averaged = prob
            optimizer.backward_and_step(loss_averaged)
            if rank == 0:
                train_printer.update(epoch + 1, loss_averaged.item(), prob_averaged.item())

        # synchronize(args['num_processes'])

    # model = SGVAE()


if __name__ == "__main__":
    main()







    # z -> p_theta(x, pi)
