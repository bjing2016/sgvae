# https://github.com/dmlc/dgl/blob/master/examples/pytorch/dgmg/model.py

NODE_ACT_DIM = 20
NODE_HIDDEN_SIZE = 40
EDGE_DIM = 60


import torch.nn as nn
import dgl
import torch
from torch.distributions import Bernoulli, Categorical
from util import *
from functools import partial

class GraphDestructor(nn.Module):
    # returns the inverse order of nodes and edges by destruction order
    def __init__(self, node_act_dim=NODE_ACT_DIM, node_hidden_size=NODE_HIDDEN_SIZE, edge_dim=EDGE_DIM):
        super(GraphDestructor, self).__init__():
        self.graph_embed = GraphEmbed(node_hidden_size)
        self.graph_prop = GraphProp(num_prop_rounds, node_hidden_size, node_act_dim, edge_dim)
        self.choose_victim_agent = ChooseVictimAgent(self.graph_prop, node_hidden_size)
    def get_log_prob(self):
        return torch.cat(self.choose_victim_agent.log_prob).sum()
    def encode(self, victims=None):
        # The graph we will work on
        self.g = dgl.DGLGraph()

        # If there are some features for nodes and edges,
        # zero tensors will be set for those of new nodes and edges.
        self.g.set_n_initializer(dgl.frame.zero_initializer)
        self.g.set_e_initializer(dgl.frame.zero_initializer)
        victim_probs = []
        victim_order = []
        remaining_nodes = list(range(g.number_of_nodes()))
        while self.g.number_of_nodes() > 1:
            self.graph_prop(g)
            victim_index, victim_prob = self.choose_victim_agent(g)
            victim_order.append(remaining_nodes.pop(victim_index))
        return self.g.nodes[0].data['hv'], victim_order, sum(victim_probs)

class ChooseVictimAgent(nn.Module):
    def __init__(self, node_hidden_size):
        super(ChooseVictimAgent, self).__init__()

        self.choose_death = nn.Linear(node_hidden_size, 1)

    def _initialize_edge_repr(self, g, src_list, dest_list):
        # For untyped edges, we only add 1 to indicate its existence.
        # For multiple edge types, we can use a one hot representation
        # or an embedding module.
        edge_repr = torch.ones(len(src_list), 1)
        g.edges[src_list, dest_list].data['he'] = edge_repr

    def forward(self, g):
        node_embeddings = g.ndata
        print("node embedding shape", node_embeddings.shape)
        death_probs = self.choose_death(node_embeddings)
        death_probs = F.softmax(death_probs, dim=1)
        dist = Categorical(death_probs)

        victim = dist.sample().item()
        victim_prob = dist.log_prob(victim)
        g.remove_nodes([victim])
        return victim, victim_prob

'''
Implementation adapted from https://docs.dgl.ai/en/latest/tutorials/models/3_generative_model/5_dgmg.html
'''
class GraphConstructor(nn.Module):
    # TODO: 
    def __init__(self, node_dim, graph_dim, act_dim, num_edge_types):
        super(GraphConstructor, self).__init__()

    def build(self, seed):
        stop = self.add_node_and_update()

        while (not stop) and (self.g.number_of_nodes() < self.v_max + 1):
            num_trials = 0
            to_add_edge = self.add_edge_or_not()
            while to_add_edge and (num_trials < g.number_of_nodes() - 1):
                self.choose_dest_and_update()
                num_trials += 1
                to_add_edge = self.add_edge_or_not()
            stop = self.add_node_and_update()

        return self.g

    def add_node_and_update(self, a=None):
        """Decide if to add a new node.
        If a new node should be added, update the graph."""
        return NotImplementedError

    def add_edge_or_not(self, a=None):
        """Decide if a new edge should be added."""
        return NotImplementedError

    def choose_dest_and_update(self, a=None):
        """Choose destination and connect it to the latest node.
        Add edges for both directions and update the graph."""
        return NotImplementedError

    def forward_train(self, actions):
        """Forward at training time. It records the probability
        of generating a ground truth graph following the actions."""
        self.prepare_for_train()

        stop = self.add_node_add_update(a=actions[sef.action_step])
        while not stop:
            to_add_edge = self.add_edge_or_not(a=actions[self.action_step])
            while to_add_edge:
                self.choose_dest_and_update(a=actions[self.action_step])
            stop = self.add_node_and_update(a=actions[self.action_step])

        return self.get_log_prob()

    def forward_inference(self):
        """Forward at inference time.
        It generates graphs on the fly."""
        return NotImplementedError

    def forward(self, actions=None):
        # The graph we will work on
        self.g = dgl.DGLGraph()

        # If there are some features for nodes and edges,
        # zero tensors will be set for those of new nodes and edges.
        self.g.set_n_initializer(dgl.frame.zero_initializer)
        self.g.set_e_initializer(dgl.frame.zero_initializer)

        if self.training:
            return self.forward_train(actions=actions)
        else:
            return self.forward_inference()

class GraphEmbed(nn.Module):
    def __init__(self, node_dim, graph_dim):
        super(GraphEmbed, self).__init__()
        self.graph_dim = graph_dim
        self.node_gating = nn.Sequential(
            nn.Linear(node_dim, 1),
            nn.Sigmoid()
        )
        self.node_to_graph = nn.Linear(node_dim, graph_dim)

    def forward(self, g):
        if g.number_of_nodes == 0:
            return torch.zeros(1, self.graph_dim)
        nodes = g.ndata['hv'] # Check this
        return (self.node_gating(nodes) * self.node_to_graph(nodes)).sum(0, keepdim=True)

class GraphProp(nn.Module):
    def __init__(self, rounds, node_dim, node_act_dim, edge_dim):
        '''
        Rounds: number of propogation steps
        '''
        super(GraphProp, self).__init__()
        self.node_act_dim = node_act_dim
        self.rounds = rounds
        
        # Each propogation step has same dims but different params
        self.message_funcs = [] # Functions transformating a cocatenation of hu, hv, xuv to a vector
        self.reduce_funcs = [] # Sums incoming messages to be activation
        self.node_update_funcs = [] # GRU cell
        for t in range(rounds):
            self.message_funcs.append(nn.Linear(2 * node_dim + edge_dim, node_act_dim))
            self.reduce_funcs.append(partial(self.dgmg_reduce, round=t))
            self.node_update_funcs.append(nn.GRUCell(self.node_act_dim, node_dim))

        # Originally ModuleLists

    def dgmg_msg(self, edges):
        return {'m' : torch.cat([edges.src['hv'], edges.dst['hv'],
                                    edges.data['he']], dim = 1)}

    def dgmg_reduce(self, nodes, round):
        #hv_old = nodes.data['hv']
        m = nodes.mailbox['m']
        #message = torch.cat([
        #    hv_old.unsqueeze(1).expand(-1, m.size(1), -1), m], dim = 2)
        node_activation = (self.message_funcs[round](m)).sum(1)

        return {'a':node_activation}

    def forward(self, g):
        if g.number_of_edges() > 0:
            for t in range(self.rounds):
                g.update_all(message_func=self.dgmg_msg, #the message func field is just the gathering step
                        reduce_func=self.reduce_funcs[t])
                g.ndata['hv'] = self.node_update_funcs[t](
                    g.ndata['a'], g.ndata['hv']
                )

class AddNode(nn.Module):
    def __init__(self, graph_embed_func, node_dim, node_act_dim):
        super(AddNode, self).__init__()

        self.graph_embed_func = graph_embed_func
        self.add_node = nn.Linear(graph_embed_func.graph_dim, 1)

        self.node_init = nn.Linear(graph_embed_func.graph_dim, node_dim)

        self.init_node_activation = torch.zeros(1, node_act_dim) # To satisfy the GRU cell

        self.log_prob = []

    def initialize_node(self, g, graph_embed):
        hv_init = self.node_init(graph_embed)
        N = g.number_of_nodes()
        g.nodes[N-1].data['hv'] = hv_init
        g.nodes[N-1].data['a'] = self.init_node_activation

    def forward(self, g, action=None):
        graph_embed = self.graph_embed_func(g)
        logit = self.add_node(graph_embed)
        prob = torch.sigmoid(logit)

        if action == None:
            action = Bernoulli(prob).sample().item()
        if action == 1: # not stop
            g.add_nodes(1)
            self.initialize_node(g, graph_embed)

        log_prob = bernoulli_action_log_prob(logit, action)
        self.log_prob.append(log_prob)
        
        return bool(action)

class AddEdges(nn.Module):
    def __init__(self, graph_embed_func, node_dim, num_edge_types):
        super(AddEdges, self).__init__()

        self.num_edge_types = num_edge_types
        self.graph_embed_func = graph_embed_func
        self.add_edge = nn.Linear(graph_embed_func.graph_dim + 2 * node_dim, num_edge_types+1) # index 0 being no edge


        # Static edge types

    def forward(self, g, action=None):
        N = g.number_of_nodes()
        graph_embed = self.graph_embed_func(g).repeat(N-1, 1)
        curr_embed = g.nodes[N-1].data['hv'].repeat(N-1, 1) # 1, node_dim
        prev_embed = g.ndata['hv'][:N-1] # N-1, node_dim


        logits = self.add_edge(torch.cat(
            [graph_embed, prev_embed, curr_embed], dim=1 # Check dims
        )) # N-1, edge_types + 1

        #node_probs = torch.sigmoid(logits, dim=0)
        if action == None:
            actions = Categorical(logits=logits).sample().numpy().tolist()
            print(actions)
            
            for idx, etype in enumerate(actions):
                if etype == 0: continue
                e_em = torch.zeros(1, self.num_edge_types)
                e_em[0,etype-1] = 1.0
                
                g.add_edges(N-1, idx)
                g.add_edges(idx, N-1)
                print('adding edge between', idx, N-1)
                g.edges[N-1, idx].data['he'] = e_em
                print(idx, etype, e_em, g.edges[idx, N-1])
                
                g.edges[idx, N-1].data['he'] = e_em
        
        log_prob = F.log_softmax(logits, dim=1).gather(1, torch.tensor(actions).long().view(-1, 1)).sum()
        self.log_prob.append(log_prob)
        return actions
            
                
                
        



        


        
if __name__ == "__main__":
    g = dgl.DGLGraph()
    # add 34 nodes into the graph; nodes are labeled from 0~33
    g.add_nodes(34)
    # all 78 edges as a list of tuples
    edge_list = [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2),
        (4, 0), (5, 0), (6, 0), (6, 4), (6, 5), (7, 0), (7, 1),
        (7, 2), (7, 3), (8, 0), (8, 2), (9, 2), (10, 0), (10, 4),
        (10, 5), (11, 0), (12, 0), (12, 3), (13, 0), (13, 1), (13, 2),
        (13, 3), (16, 5), (16, 6), (17, 0), (17, 1), (19, 0), (19, 1),
        (21, 0), (21, 1), (25, 23), (25, 24), (27, 2), (27, 23),
        (27, 24), (28, 2), (29, 23), (29, 26), (30, 1), (30, 8),
        (31, 0), (31, 24), (31, 25), (31, 28), (32, 2), (32, 8),
        (32, 14), (32, 15), (32, 18), (32, 20), (32, 22), (32, 23),
        (32, 29), (32, 30), (32, 31), (33, 8), (33, 9), (33, 13),
        (33, 14), (33, 15), (33, 18), (33, 19), (33, 20), (33, 22),
        (33, 23), (33, 26), (33, 27), (33, 28), (33, 29), (33, 30),
        (33, 31), (33, 32)]
    # add edges two lists of nodes: src and dst
    src, dst = tuple(zip(*edge_list))
    g.add_edges(src, dst)
    # edges are directional in DGL; make them bi-directional
    g.add_edges(dst, src)



    print('compiled without issue')

    ## Notes
    # DGLGraph object:
    #   nodes(): tensor of nodes. Must be 0...N-1
    #   edges(): tensor of src, tensor of dest
    #   add_nodes(number of nodes to add)
    #   add_edges(src list, dest list)
    #   nodes[idx]: node object with dictionary .data, also accessible as tensor .ndata[key]
    #   edges[src, dest].data similarly, also as .edges[[src list], [dest list]].data
    #   g.edata is not intuitive sorted, so be careful
    #   Careful when setting schemes
    #
