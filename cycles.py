from cycle_dataset import CycleDataset
import torch
import torch.optim as optim

import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.utils.data as utils
from models import *
from tqdm import tqdm


def train(num_epochs=200):
    sgvae = SGVAE(rounds=2,
                    node_dim=5,
                    msg_dim=6,
                    edge_dim=3,
                    graph_dim=7,
                    num_node_types=2,
                    lamb=1)
    train = CycleDataset('cycles/train.cycles')
    val = CycleDataset('cycles/val.cycles')

    trainLoader = utils.DataLoader(train, batch_size=1, shuffle=True, num_workers=0,
                             collate_fn=train.collate_single)

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
        loss_sum = 0
        for g in tqdm(trainLoader, desc="[{}]".format(epoch)):
            loss = sgvae.loss(g)
            loss_sum += loss
        loss_sum /= len(trainLoader)
        loss_sum.backward()
        optimizer.step()
        print(loss_sum)
    print("Saving to {}.params".format(epoch))
    torch.save(sgvae.state_dict(), 'params/{}.params'.format(epoch))

def eval(epoch):
    sgvae = SGVAE(rounds=2,
                    node_dim=5,
                    msg_dim=6,
                    edge_dim=3,
                    graph_dim=7,
                    num_node_types=2,
                    lamb=1)
    sgvae.load_state_dict(torch.load('params/{}.params'.format(epoch)))
    sgvae.eval()

    graph = sgvae.generate
    print(graph)

def main():
    if sys.argv[1] == 'train':
        train()
    else:
        eval(sys.argv[2])
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
