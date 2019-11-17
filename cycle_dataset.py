#import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm
import random
import pickle
import dgl
import torch
import numpy as np


from torch.utils.data import Dataset

def create_cycle_with_size(size):
    g = dgl.DGLGraph()
    g.add_nodes(size)
    g.ndata['out'] = torch.FloatTensor(np.tile([0, 1], [size, 1]))
    src, dest = np.arange(size), np.roll(np.arange(size), -1)
    g.add_edges(src, dest)
    g.add_edges(dest, src)
    g.edata['he'] = torch.FloatTensor(np.tile([1, 0, 0], [2*size, 1]))
    return g

def generate_dataset(minSize, maxSize, nSamples, fname):
    samples = []
    for i in tqdm(range(nSamples)):
        size = random.randint(minSize, maxSize)
        samples.append(create_cycle_with_size(size))

    with open(fname, 'wb') as f:
        pickle.dump(samples, f)

    print("Wrote {} examples to {}".format(nSamples, fname))

class CycleDataset(Dataset):
    def __init__(self, fname):
        super(CycleDataset, self).__init__()

        with open(fname, 'rb') as f:
            self.dataset = pickle.load(f)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

    def collate_single(self, batch):
        assert len(batch) == 1, 'Currently we do not support batched training'
        return batch[0]

    def collate_batch(self, batch):
        return batch


class CycleModelEvaluation:
    def __init__(self, v_min, v_max, dir):
        super(CycleModelEvaluation, self).__init__()

        self.v_min = v_min
        self.v_max = v_max

        self.dir = dir

    def rollout_and_examine(self, model, num_samples):
        assert not model.training, 'You need to call model.eval().'

        num_total_size = 0
        num_valid_size = 0
        num_cycle = 0
        num_valid = 0
        plot_times = 0
        adj_lists_to_plot = []

        for i in range(num_samples):
            sampled_graph = model()
            if isinstance(sampled_graph, list):
                # When the model is a batched implementation, a list of
                # DGLGraph objects is returned. Note that with model(),
                # we generate a single graph as with the non-batched
                # implementation. We actually support batched generation
                # during the inference so feel free to modify the code.
                sampled_graph = sampled_graph[0]

            sampled_adj_list = dglGraph_to_adj_list(sampled_graph)
            adj_lists_to_plot.append(sampled_adj_list)

            graph_size = sampled_graph.number_of_nodes()
            valid_size = (self.v_min <= graph_size <= self.v_max)
            cycle = is_cycle(sampled_graph)

            num_total_size += graph_size

            if valid_size:
                num_valid_size += 1

            if cycle:
                num_cycle += 1

            if valid_size and cycle:
                num_valid += 1

            if len(adj_lists_to_plot) >= 4:
                plot_times += 1
                fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2)
                axes = {0: ax0, 1: ax1, 2: ax2, 3: ax3}
                for i in range(4):
                    nx.draw_circular(nx.from_dict_of_lists(adj_lists_to_plot[i]),
                                     with_labels=True, ax=axes[i])

                plt.savefig(self.dir + '/samples/{:d}'.format(plot_times))
                plt.close()

                adj_lists_to_plot = []

        self.num_samples_examined = num_samples
        self.average_size = num_total_size / num_samples
        self.valid_size_ratio = num_valid_size / num_samples
        self.cycle_ratio = num_cycle / num_samples
        self.valid_ratio = num_valid / num_samples

    def write_summary(self):

        def _format_value(v):
            if isinstance(v, float):
                return '{:.4f}'.format(v)
            elif isinstance(v, int):
                return '{:d}'.format(v)
            else:
                return '{}'.format(v)

        statistics = {
            'num_samples': self.num_samples_examined,
            'v_min': self.v_min,
            'v_max': self.v_max,
            'average_size': self.average_size,
            'valid_size_ratio': self.valid_size_ratio,
            'cycle_ratio': self.cycle_ratio,
            'valid_ratio': self.valid_ratio
        }

        model_eval_path = os.path.join(self.dir, 'model_eval.txt')

        with open(model_eval_path, 'w') as f:
            for key, value in statistics.items():
                msg = '{}\t{}\n'.format(key, _format_value(value))
                f.write(msg)

        print('Saved model evaluation statistics to {}'.format(model_eval_path))


class CyclePrinting(object):
    def __init__(self, num_epochs, num_batches):
        super(CyclePrinting, self).__init__()

        self.num_epochs = num_epochs
        self.num_batches = num_batches
        self.batch_count = 0

    def update(self, epoch, metrics):
        self.batch_count = (self.batch_count) % self.num_batches + 1

        msg = 'epoch {:d}/{:d}, batch {:d}/{:d}'.format(epoch, self.num_epochs,
                                                        self.batch_count, self.num_batches)
        for key, value in metrics.items():
            msg += ', {}: {:4f}'.format(key, value)
        print(msg)

if __name__ == "__main__":
    minSize, maxSize, nSamples = [int(x) for x in sys.argv[1:4]]
    fname = sys.argv[4]
    generate_dataset(minSize, maxSize, nSamples, fname)
