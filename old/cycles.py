import sys
from pi import f
import matplotlib.pyplot as plt
import networkx as nx
from cycle_dataset import CycleDataset
import torch
#torch.set_num_threads(1)
import torch.optim as optim

import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.utils.data as utils
from models import *
from tqdm import tqdm, trange
from copy import deepcopy


def is_cycle(g):
    size = g.number_of_nodes()

    if size < 3:
        return False

    node = 0
    prev = None
    lastNode = None
    seen = set()
    for i in range(size):
        print(node)
        neighbors = g.successors(node)
        if len(neighbors) != 2:
            return False

        if prev is None:
            node = g.successors(node)[0]
        else:
            for poss in neighbors:
                if poss == prev: continue
                elif poss in seen: return False
                else:
                    seen.add(poss)
                    node = poss
        if node == 0: return False
        prev = node
    if 0 not in g.successors(node): return False

    return True



def train(num_epochs=200):
    num_epochs = int(num_epochs)
    
    sgvae = SGVAE(rounds=3,
                    node_dim=3,
                    msg_dim=6,
                    edge_dim=3,
                    graph_dim=30,
                    num_node_types=2,
                    lamb=1)

    destructor = sgvae.encoder
    constructor = sgvae.decoder

    trainData = CycleDataset('cycles/train.cycles')
    g = trainData[0]
    
    
    #z, pi, __ = destructor(deepcopy(g))
    #pi = range(7)
    #print(pi)
    optimizer = optim.SGD(sgvae.parameters(), lr=0.01)
    t = trange(18000)

    for i in t:
        optimizer.zero_grad()
        #z, pi, log_qzpi = destructor(deepcopy(g))
        #_, prob = constructor(z, pi=pi, target=g)


        loss, genGraph, z, log_qzpi, prob, unldr = sgvae.loss(g, return_graph=True)
        #f.write(str(pi))# (z.detach().numpy(), pi, float(log_qzpi), float(prob))))
        #f.write('\n')
        #f.flush()
        (loss).backward(retain_graph=False)
        #(-log_qzpi-prob).backward(retain_graph=False)
        optimizer.step()
        s = '{:.4f} {:.4f}'.format(float(log_qzpi), float(prob))
        sprime = '{:.4f}'.format(float(unldr))
        f.write(s + ' ' + sprime + '\n')
        f.flush()
        t.set_description('{:.4f}'.format(float(unldr)))
        
        if i % 100 == 0:
            new = sgvae.generate()
            plt.clf()
            nx.draw(new.to_networkx())
            plt.savefig('outputs/{}.png'.format(i))
        
    exit()
    valData = CycleDataset('cycles/val.cycles')

    optimizer = optim.SGD(sgvae.parameters(), lr=0.01, momentum=0.9)

    # for g in trainLoader:
    #     print(g)
    #     break

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        print("Epoch", epoch)
        if epoch % 5 == 0 and epoch != 0:
            print("Saving to {}.params".format(epoch))
            torch.save(sgvae.state_dict(), 'params/{}.params'.format(epoch))
            eval(epoch, writeFile=True, z_value=z)
        loss_sum = 0
        for g in tqdm(trainLoader, desc="[{}]".format(epoch)):
            loss, genGraph, z = sgvae.loss(g, return_graph=True)
            loss_sum += loss
        loss_sum /= len(trainLoader)
        loss_sum.backward()
        optimizer.step()
        print(loss_sum)
    print("Saving to {}.params".format(epoch))
    torch.save(sgvae.state_dict(), 'params/{}.params'.format(epoch))

def eval(epoch, writeFile=False, z_value=None, calc_cycle=False):
    sgvae = SGVAE(rounds=6,
                    node_dim=5,
                    msg_dim=6,
                    edge_dim=3,
                    graph_dim=30,
                    num_node_types=2,
                    lamb=1)
    sgvae.load_state_dict(torch.load('params/{}.params'.format(epoch)))
    sgvae.eval()

    graph = sgvae.generate(z_value=z_value)
    print(graph)
    plt.clf()
    nx.draw(graph.to_networkx())
    if writeFile:
        plt.savefig('outputs/{}.png'.format(epoch))
    else:
        plt.show()
    if calc_cycle:
        NUM = 3000
        out = 0
        for i in range(NUM):
            graph = sgvae.generate(z_value=z_value)
            out += is_cycle(graph)
            # if is_cycle(graph):
            #     plt.clf()
            #     nx.draw(graph.to_networkx())
            #     plt.show()
        print(out/NUM)


def main():
    if sys.argv[1] == 'train':
        train(*sys.argv[2:])
    elif sys.argv[1] == 'vis':
        trainData = CycleDataset('cycles/five_train.cycles')
        for x in trainData:
            break
        assert is_cycle(x)
        nx.draw(x.to_networkx())
        plt.show()
    else:
        eval(sys.argv[2], calc_cycle=True)
#
# def is_valid(g):
#     # Check if g is a cycle having 10-20 nodes.
#     def _get_previous(i, v_max):
#         if i == 0:
#             return v_max
#         else:
#             return i - 1
#
#     def _get_next(i, v_max):
#         if i == v_max:
#             return 0
#         else:
#             return i + 1
#
#     size = g.number_of_nodes()
#
#     if size < 10 or size > 20:
#         return False
#
#     for node in range(size):
#         neighbors = g.successors(node)
#
#         if len(neighbors) != 2:
#             return False
#
#         if _get_previous(node, size - 1) not in neighbors:
#             return False
#
#         if _get_next(node, size - 1) not in neighbors:
#             return False
#
#     return True
#
# num_valid = 0
# for i in range(100):
#     g = model()
#     num_valid += is_valid(g)
#
# del model
# print('Among 100 graphs generated, {}% are valid.'.format(num_valid))
#
# def main():
#     dataset = MoleculeDataset('ChEMBL', 'canonical', ['train', 'val'])
#     train_loader = DataLoader(dataset.train_set, batch_size=args['batch_size'],
#                               shuffle=True, collate_fn=dataset.collate)
#     val_loader = DataLoader(dataset.val_set, batch_size=args['batch_size'],
#                             shuffle=True, collate_fn=dataset.collate)
#     model = SGVAE(atom_types=dataset.atom_types,
#                   bond_types=dataset.bond_types,
#                   node_hidden_size=args['node_hidden_size'],
#                   num_prop_rounds=args['num_propagation_rounds'],
#                   dropout=args['dropout'])
#     if args['num_processes'] == 1:
#         from utils import Optimizer
#         optimizer = Optimizer(args['lr'], Adam(model.parameters(), lr=args['lr']))
#     else:
#         from utils import MultiProcessOptimizer
#         optimizer = MultiProcessOptimizer(args['num_processes'], args['lr'],
#                                           Adam(model.parameters(), lr=args['lr']))
#
#     if rank == 0:
#         t2 = time.time()
#     best_val_prob = 0
#
#     # Training
#     for epoch in range(args['nepochs']):
#         model.train()
#         if rank == 0:
#             print('Training')
#
#         for i, data in enumerate(train_loader):
#             log_prob = model(actions=data, compute_log_prob=True)
#             prob = log_prob.detach().exp()
#
#             loss_averaged = - log_prob
#             prob_averaged = prob
#             optimizer.backward_and_step(loss_averaged)
#             if rank == 0:
#                 train_printer.update(epoch + 1, loss_averaged.item(), prob_averaged.item())
#
#         # synchronize(args['num_processes'])
#
#     # model = SGVAE()


if __name__ == "__main__":
    main()







    # z -> p_theta(x, pi)
