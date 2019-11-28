from cycle_dataset import CycleDataset
from evaluation import *

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



trainData = CycleDataset('datasets/cycles.pkl')
train('cycles', trainData, is_cycle, batch_size=10, num_epochs=20000, stop_file='stopfile')

'''Interpolation
sgvae = RESTORE PARAMS
import cycle_dataset
x1 = cycle_dataset.create_cycle_with_size(5)
x2 = cycle_dataset.create_cycle_with_size(15)
evaluate(sgvae, is_cycle, x1, x2, lambda g: g.number_of_nodes())
'''