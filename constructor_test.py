import sys
import matplotlib.pyplot as plt
import networkx as nx
from cycle_dataset import CycleDataset
import torch
import torch.optim as optim

import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.utils.data as utils
from models import *
from tqdm import tqdm


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

class ConstructorOnly(nn.Module):
    def __init__(self, rounds, node_dim, msg_dim, edge_dim, graph_dim, num_node_types, lamb):
        super(SGVAE, self).__init__()
        self.lamb = lamb
        self.decoder = GraphConstructor(node_dim=node_dim,
                                        graph_dim=graph_dim,
                                        msg_dim=msg_dim,
                                        num_edge_types=edge_dim,
                                        num_prop_rounds=rounds,
                                        num_node_types=num_node_types)
        self.z_prior = (nn.Parameter(torch.zeros([node_dim]), requires_grad=False),
                        torch.diag(nn.Parameter(torch.ones([node_dim]), requires_grad=False)))
        self.node_dim = node_dim



    def loss(self, x, return_graph=False):
        # note that nothing is batched !
        target = deepcopy(x)
        orig = deepcopy(x)
        # z, pi, log_qzpi = self.encoder(orig)
        # print("pi is", pi)
        # print("z, l", z, log_qzpi)
        #print(pi)
        # z     := vector of dimension z_dim
        # pi    := vector of [[idx] + [order]]
        # log_q := scalar
        z = torch.zeros(self.node_dim)
        pi = torch.FloatTensor([0, 1, 2, 3, 4])
        # log_pz = MultivariateNormal(*self.z_prior).log_prob(z) # Can we do this in init?
        genGraph, log_px = self.decoder(z, pi=pi, target=target)

        # unldr = (log_px + log_pz - log_qzpi).detach()
        # print(unldr)
        # loss = -(log_qzpi * (unldr - 1) + self.lamb * log_px)
        loss = -log_px

        # unldr = unldr.detach()
        # loss = unldr - 1/()
        # print(*[float(x) for x in [loss, unldr, log_qzpi, log_pz, log_px]])
        # print("loss:", loss)
        # print("unldr:", unldr)
        # print("log_qzpi:", log_qzpi)
        # print("log_pz:", log_pz)
        # print("log_px:", log_px)
        if return_graph:
            return loss, genGraph, z
        else:
            return loss

    def generate(self, numToGenerate=1, z_value=None):
        # if z_value is None:
        #     z_dist = torch.distributions.multivariate_normal.MultivariateNormal(*self.z_prior)
        #     z_value = z_dist.sample([numToGenerate])
        z_value = torch.zeros(self.node_dim)
        graph, _ = self.decoder(z_value)
        # print(graph.ndata['hv'].shape)
        return graph


def train(num_epochs=200):
    num_epochs = int(num_epochs)
    sgvae = ConstructorOnly(rounds=6,
                    node_dim=5,
                    msg_dim=6,
                    edge_dim=3,
                    graph_dim=30,
                    num_node_types=2,
                    lamb=1)
    trainData = CycleDataset('cycles/train.cycles')
    valData = CycleDataset('cycles/val.cycles')

    trainLoader = utils.DataLoader(trainData, batch_size=1, shuffle=False, num_workers=0,
                             collate_fn=trainData.collate_single)

    optimizer = optim.SGD(sgvae.parameters(), lr=0.001, momentum=0.9)


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
