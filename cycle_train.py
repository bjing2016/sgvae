from cycle_dataset import CycleDataset
from evaluation import *
from copy import deepcopy
import sys

def is_cycle(g):
    size = g.number_of_nodes()

    if size < 3:
        return False

    node = 0
    prev = None
    lastNode = None
    seen = set()
    for i in range(size):
        #print(node)
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

def better_is_cycle(g):
    #g = deepcopy(g)
    node = 0
    seen = set()
    neighbors = g.successors(node)
    if len(neighbors) != 2:
        return False
    #g.remove_nodes([node])
    seen.add(node)
    node = neighbors[0]
    while True:
        #outputfile.write(str(seen) + 'SEEN\n')
        #outputfile.flush()
        initneighbors = set(int(f) for f in g.successors(node))
        if len(initneighbors) != 2: return False
        neighbors = set(int(f) for f in g.successors(node)) - seen
        #outputfile.write(str(neighbors) + 'NEIGHBORS\n')
        #outputfile.flush()
        
        if len(neighbors) != 1: break
        #g.remove_nodes([node])
        seen.add(int(node))
        node = list(neighbors)[0]
    return g.number_of_nodes() - len(seen) == 1 #and 0 in [int(f) for f in g.successors(node)] and len(g.successors(node)) == 2
    #return g.number_of_nodes() == 1
        



epoch = 60

params = 'train/cycles{}.params'.format(epoch)
sgvae = torch.load(params)
#sgvae = None
optimizer = torch.load('optimizer')
#optimizer = None
trainData = CycleDataset('datasets/cycles.pkl')
sgvae, optim = train('cycles', trainData, better_is_cycle, batch_size=10, num_epochs=epoch+21, sgvae=sgvae, start_epoch=epoch+1, optimizer=optimizer)#torch.load('optimizer'))
torch.save(optim, 'optimizer')

'''Interpolation
sgvae = RESTORE PARAMS
import cycle_dataset
x1 = cycle_dataset.create_cycle_with_size(5)
x2 = cycle_dataset.create_cycle_with_size(15)
evaluate(sgvae, is_cycle, x1, x2, lambda g: g.number_of_nodes())
'''
