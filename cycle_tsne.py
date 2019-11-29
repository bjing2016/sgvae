from cycle_dataset import *
from evaluation import *
import tqdm
from copy import deepcopy
from sklearn.manifold import TSNE
import matplotlib.cm as cm

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

def analyze(sgvae):
    zs = []
    N_DUP = 30
    for i in trange(5, 20):
        g = create_cycle_with_size(i)
        for _ in range(N_DUP):
            z = sgvae.encoder(deepcopy(g))[0]
            zs.append(z)
    zs = torch.cat(zs, dim=0)
    zs_embedded = TSNE(n_components=2).fit_transform(zs.detach().numpy())
    print(zs_embedded.shape)

    colors = cm.rainbow(np.linspace(0, 1, 15))
    
    for i in range(15):
        plt.scatter(zs_embedded[i*N_DUP:(i+1)*N_DUP,0], zs_embedded[i*N_DUP:(i+1)*N_DUP,1], label=str(i+5), color=colors[i])
    plt.legend()
    plt.show()

    
for epoch in range(60, 70, 10):
    params = 'train/cycles{}.params'.format(epoch)
    sgvae = torch.load(params)
    #frac_acceptable = count_acceptable(sgvae, is_cycle)
    #print(frac_acceptable)
    analyze(sgvae)
    
