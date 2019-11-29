import matplotlib.pyplot as plt
import torch
from models import SGVAE
import torch.utils.data as utils
import networkx as nx
import torch.optim as optim
from tqdm import tqdm, trange
from output import outputfile
import numpy as np


def train(name, dataset, accept_func, batch_size=10, num_epochs=1000, sgvae=None, start_epoch=0, optimizer=None):
    '''
    |dataset| 100 examples
    |accept_func|: DGLGraph -> bool
    |feature_func|: DGLGraph -> float
    |stop_file| write "stop" to this file to stop
    '''
    if sgvae == None:
        sgvae = SGVAE(rounds=2,
                        node_dim=5,             # TODO: PLEASE DO NOT FEED IN DATA WITH MORE THAN ONE EDGE OR NODE TYPE!!!!!
                        msg_dim=6,
                        edge_dim=1,
                        graph_dim=30,
                        num_node_types=1,
                        lamb=1)

    trainLoader = utils.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0,
                             collate_fn=dataset.collate_single)
    if optimizer == None: optimizer = optim.Adam(sgvae.parameters(), lr=0.01)
    
    for epoch in range(start_epoch, num_epochs):
        t = tqdm(trainLoader)
        loss_sum = 0
        unldr_sum = 0
        for i, g in enumerate(t):
            loss, genGraph, z, log_qzpi, log_px, unldr = sgvae.loss(g, return_graph=True)
            loss_sum += loss
            t.set_description("{:.3f}".format(float(unldr)))
            unldr_sum += float(unldr)
            if (i + 1) % batch_size == 0:
                optimizer.zero_grad()
                loss_sum /= batch_size
                loss_sum.backward()
                optimizer.step()
                loss_sum = 0
        
        new = sgvae.generate()
        nx.draw(new.to_networkx())
        plt.savefig('train/{}{}.png'.format(name, epoch))
        plt.clf()

        frac_acceptable = count_acceptable(sgvae, accept_func)
        outputfile.write('{},{},{},{}\n'.format(name, epoch, frac_acceptable, unldr_sum/len(dataset)))
        outputfile.flush()

        if epoch % 10 == 0 or epoch == (num_epochs - 1):
            print("Saving to {}.params".format(epoch))
            torch.save(sgvae, 'train/{}{}.params'.format(name, epoch))
            
    return sgvae, optimizer

yescycles = 0
def count_acceptable(sgvae, accept_func):
    acceptable = 0
    for i in trange(100):
        g = sgvae.generate()
        if accept_func(g): 
            acceptable += 1
            global yescycles
            #save(g, 'yescycles/{}.png'.format(yescycles))
            
            yescycles+=1
        else: 
            pass
            #save(g, 'notcycles/{}.png'.format(notcycles))
            #outputfile.write('{}.png IS NOT A CYCLE\n-----------\n'.format(notcycles))
            #outputfile.flush()
    return acceptable / 100

notcycles = 0
def save(g, path):
    global notcycles
    notcycles+=1
    nx.draw(g.to_networkx())
    plt.savefig(path)
    plt.clf()
    plt.close()

def evaluate(sgvae, accept_func, x1, x2, feature_func):
    destructor = sgvae.encoder
    constructor = sgvae.decoder
    z1 = z2 = 0
    for i in range(100):
        z1 += destructor(x1)[0]
        z2 += destructor(x2)[0]
    z1 /= 100
    z2 /= 100

    for p in np.arange(0, 1.001, 0.02):
        z = p*z1 + (1-p)*z2
        total = 0
        acceptable = 0
        gs = []
        features = []
        while acceptable < 100:
            total += 1
            g = constructor(z)
            if accept_func(g):
                acceptable += 1
                gs.append(g)
                features.append(feature_func(g))
        s = '{},{},{}'.format(p, total, feature_func)
        print(s)
        outputfile.write(s)
        outputfile.flush()
