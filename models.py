# https://github.com/dmlc/dgl/blob/master/examples/pytorch/dgmg/model.py

NODE_ACT_DIM = 20
NODE_HIDDEN_SIZE = 40
EDGE_DIM = 60


import torch.nn as nn
import dgl
import torch
from torch.distributions import Bernoulli, Categorical, MultivariateNormal
from util import *
from functools import partial
from copy import deepcopy

MAX_NUM_NODES = 10

class SGVAE(nn.Module):
    def __init__(self, rounds, node_dim, msg_dim, edge_dim, graph_dim, num_node_types, lamb):
        super(SGVAE, self).__init__()
        self.lamb = lamb
        self.encoder = GraphDestructor(rounds=rounds,
                                        node_dim=node_dim,
                                        msg_dim=msg_dim,
                                        edge_dim=edge_dim,
                                        num_node_types=num_node_types)
        self.decoder = GraphConstructor(node_dim=node_dim,
                                        graph_dim=graph_dim,
                                        msg_dim=msg_dim,
                                        num_edge_types=edge_dim,
                                        num_prop_rounds=rounds,
                                        num_node_types=num_node_types)
        self.z_prior = (nn.Parameter(torch.zeros([node_dim]), requires_grad=False),
                        torch.diag(nn.Parameter(torch.ones([node_dim]), requires_grad=False)))



    def loss(self, x, return_graph=False):
        # note that nothing is batched !
        target = deepcopy(x)
        orig = deepcopy(x)
        z, pi, log_qzpi = self.encoder(orig)
        print("pi is", pi)
        # print("z, l", z, log_qzpi)
        #print(pi)
        # z     := vector of dimension z_dim
        # pi    := vector of [[idx] + [order]]
        # log_q := scalar
        log_pz = MultivariateNormal(*self.z_prior).log_prob(z) # Can we do this in init?
        #log_pz = log_gaussian(z, self.z_prior)
        # log_pz := scalar
        #print(x.number_of_nodes())
        genGraph, log_px = self.decoder(z, pi=pi, target=target)
        # genGraph := Graph
        # log_px := scalar

        # unldr = (log_px.detach() + log_pz.detach() - log_qzpi) # unnormalized log-density ratio ?
        # loss = -(unldr + log_qzpi * unldr.detach() + self.lamb * log_px)

        unldr = (log_px + log_pz - log_qzpi).detach()
        print(unldr)
        loss = -(log_qzpi * (unldr - 1) + self.lamb * log_px)

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
        if z_value is None:
            z_dist = torch.distributions.multivariate_normal.MultivariateNormal(*self.z_prior)
            z_value = z_dist.sample([numToGenerate])
        graph, _ = self.decoder(z_value)
        # print(graph.ndata['hv'].shape)
        return graph

class GraphDestructor(nn.Module):
    # returns the inverse order of nodes and edges by destruction order
    def __init__(self, rounds, node_dim, msg_dim, edge_dim, num_node_types):
        super(GraphDestructor, self).__init__()
        self.initial_encoder = nn.Linear(num_node_types, node_dim)
        self.graph_prop = GraphProp(rounds, node_dim, msg_dim, edge_dim)
        self.choose_victim_agent = ChooseVictimAgent(node_dim)

    def get_log_prob(self):
        return torch.cat(self.choose_victim_agent.log_prob).sum()

    def forward(self, g):
        # The graph we will work on
        # print(g.ndata)
        g.ndata['hv'] = self.initial_encoder(g.ndata['out'].float())
        victim_probs = []
        victim_order = []
        remaining_nodes = list(range(g.number_of_nodes()))
        while g.number_of_nodes() > 1:
            self.graph_prop(g)
            victim_index, victim_prob = self.choose_victim_agent(g)
            victim_order.append(remaining_nodes.pop(victim_index))
            victim_probs.append(victim_prob)
        victim_order.append(remaining_nodes[0])
        # print("vp sum", sum(victim_probs))
        return g.nodes[0].data['hv'], victim_order[::-1], sum(victim_probs)

class ChooseVictimAgent(nn.Module):
    def __init__(self, node_hidden_size):
        super(ChooseVictimAgent, self).__init__()

        self.choose_death = nn.Linear(node_hidden_size, 1)

    ''' Commenting this out since it seems unused
    def _initialize_edge_repr(self, g, src_list, dest_list):
        # For untyped edges, we only add 1 to indicate its existence.
        # For multiple edge types, we can use a one hot representation
        # or an embedding module.
        edge_repr = torch.ones(len(src_list), 1)
        g.edges[src_list, dest_list].data['he'] = edge_repr '''

    def forward(self, g):
        node_embeddings = g.ndata['hv']
        #print("node embedding shape", node_embeddings.shape)
        death_probs = self.choose_death(node_embeddings)
        death_probs = F.softmax(death_probs, dim=1)
        dist = Categorical(death_probs.view(-1))
        victim = dist.sample()
        victim_prob = dist.log_prob(victim)
        g.remove_nodes([victim])
        return victim, victim_prob

'''
Implementation adapted from https://docs.dgl.ai/en/latest/tutorials/models/3_generative_model/5_dgmg.html
'''
class GraphConstructor(nn.Module):
    # TODO: node_embed -> edge types
    def __init__(self, node_dim, graph_dim, msg_dim, num_edge_types, num_prop_rounds, num_node_types):
        super(GraphConstructor, self).__init__()

        self.graph_embed_func = GraphEmbed(node_dim, graph_dim)
        self.node_adder = AddNode(self.graph_embed_func, node_dim, msg_dim)
        self.edge_adder = AddEdges(self.graph_embed_func, node_dim, num_edge_types)
        self.prop_func = GraphProp(num_prop_rounds, node_dim, msg_dim, num_edge_types)
        self.num_node_types = num_node_types
        self.node_type_extractor = nn.Linear(node_dim, num_node_types)

    def forward(self, z, pi=None, target=None):
        '''
        z: dims (1, node_embed_dim)
        pi: [n_1, n_2...] order in which graph should be constructed.
            Also serves as new (reconstructed) index -> original index lookup
        returns DGLGraph with types in ndata['out'], NOT ndata['hv']
        '''
        g = dgl.DGLGraph()
        g.add_nodes(1)

        if target != None:
            N = target.number_of_nodes()
            assert(N == len(pi))
            reverse_pi = [0]*N # old index -> new index lookup
            for new, old in enumerate(pi):
                reverse_pi[old] = new
            edges = [[0]*N for _ in range(N)]
            srcs, dsts = target.edges()
            srcs, dsts = srcs.numpy(), dsts.numpy()
            for (src, dst) in list(zip(srcs, dsts)):
                newsrc = reverse_pi[src]
                newdst = reverse_pi[dst]
                assert(newsrc != newdst)
                if newsrc < newdst:
                    edges[newdst][newsrc] =  int(torch.argmax(target.edges[src, dst].data['he'].view(-1))) + 1

        g.ndata['hv'] = z
        log_prob = []
        num_nodes = 1
        for i in range(MAX_NUM_NODES-1):
            self.prop_func(g)
            node_action = num_nodes < len(pi) if pi != None else None
            node_added, log_prob_entry = self.node_adder(g, node_action)
            log_prob.append(log_prob_entry)
            if not node_added: break
            num_nodes += 1
            edge_action = edges[num_nodes-1][:num_nodes-1] if target != None else None
            #print('has', num_nodes, 'nodes')
            #print('edge action', edge_action)
            _, log_prob_entry = self.edge_adder(g, edge_action) # edge_added, log_prob
            log_prob.append(log_prob_entry)
            # print(node_added)
            # print(num_nodes, log_prob_entry)

        type_logits = self.node_type_extractor(g.ndata['hv'])
        if target == None:
            types = Categorical(logits=type_logits).sample() # Tensor of indices
        else:
            types = target.ndata['out'].argmax(dim=1)[pi] # Argmax for each node
        out = torch.zeros(g.number_of_nodes(), self.num_node_types)
        #print(out.shape, types.shape)
        g.ndata['out'] = out.scatter_(1, types.long().view(-1, 1), 1)
        node_type_log_prob = F.log_softmax(type_logits, dim=1).gather(1, types.long().view(-1, 1)).sum()
        log_prob.append(node_type_log_prob)

        return g, sum(log_prob)



class GraphEmbed(nn.Module):
    def __init__(self, node_dim, graph_dim):
        '''
        node_dim: dimension of node embedding
        graph_dim: dimension of graph embedding
        '''
        super(GraphEmbed, self).__init__()
        self.graph_dim = graph_dim
        self.node_gating = nn.Sequential(
            nn.Linear(node_dim, 1),
            nn.Sigmoid()
        )
        self.node_to_graph = nn.Sequential(
            nn.Linear(node_dim, node_dim),
            nn.Dropout(p=0.5),
            nn.Linear(node_dim, graph_dim)
        )

    def forward(self, g):
        '''
        You should not actually ever call this module
        '''
        if g.number_of_nodes == 0:
            return torch.zeros(1, self.graph_dim)
        nodes = g.ndata['hv'] # Check this
        return (self.node_gating(nodes) * self.node_to_graph(nodes)).sum(0, keepdim=True)

class GraphProp(nn.Module):
    def __init__(self, rounds, node_dim, node_act_dim, edge_dim):
        '''
        rounds: number of propogation steps
        node_dim: number of propogation steps
        node_act_dim: message dimension
        '''
        super(GraphProp, self).__init__()
        self.node_act_dim = node_act_dim
        self.rounds = rounds

        # Each propogation step has same dims but different params
        self.message_funcs = [] # Functions transformating a cocatenation of hu, hv, xuv to a vector
        self.reduce_funcs = [] # Sums incoming messages to be activation
        self.node_update_funcs = [] # GRU cell
        for t in range(rounds):
            self.message_funcs.append(nn.Sequential(nn.Linear(2 * node_dim + edge_dim, node_act_dim), nn.Identity()))
            self.reduce_funcs.append(partial(self.dgmg_reduce, round=t))
            self.node_update_funcs.append(nn.GRUCell(self.node_act_dim, node_dim))

        # Originally ModuleLists

    def dgmg_msg(self, edges):
        '''
        Forms messages from edge data
        '''
        return {'m' : torch.cat([edges.src['hv'], edges.dst['hv'],
                                    edges.data['he']], dim = 1)}

    def dgmg_reduce(self, nodes, round):
        '''
        Reduces set of inbound messages to single vector
        '''
        #hv_old = nodes.data['hv']
        m = nodes.mailbox['m']
        #message = torch.cat([
        #    hv_old.unsqueeze(1).expand(-1, m.size(1), -1), m], dim = 2)
        node_activation = (self.message_funcs[round](m)).sum(1)

        return {'a':node_activation}

    def forward(self, g):
        '''
        Propogates
        '''
        if g.number_of_edges() > 0:
            for t in range(self.rounds):
                g.update_all(message_func=self.dgmg_msg, #the message func field is just the gathering step
                        reduce_func=self.reduce_funcs[t])
                g.ndata['hv'] = self.node_update_funcs[t](
                    g.ndata['a'], g.ndata['hv']
                )

class AddNode(nn.Module):
    def __init__(self, graph_embed_func, node_dim, node_act_dim):
        '''
        graph_embed_func: func of type GraphEmbed module above
        node_dim: node embedding dimension
        node_act_dim: message dimension
        '''
        super(AddNode, self).__init__()

        self.graph_embed_func = graph_embed_func
        self.add_node = nn.Linear(graph_embed_func.graph_dim, 1)

        self.node_init = nn.Sequential(nn.Linear(graph_embed_func.graph_dim, node_dim), nn.Identity())

        self.init_node_activation = torch.zeros(1, node_act_dim) # To satisfy the GRU cell

        self.log_prob = []

    def initialize_node(self, g, graph_embed):
        '''
        Don't call this.
        Initializes a new node based on the graph embedding.
        '''
        hv_init = self.node_init(graph_embed)
        N = g.number_of_nodes()
        g.nodes[N-1].data['hv'] = hv_init
        g.nodes[N-1].data['a'] = self.init_node_activation

    def forward(self, g, action=None):
        '''
        If action==None, decides wheter or not to add a node.
        Returns true/false, log_prob
        action needs to be True (add node) or False (not add)
        '''
        graph_embed = self.graph_embed_func(g)
        logit = self.add_node(graph_embed)
        prob = torch.sigmoid(logit)


        if action == None:
            # print(prob, graph_embed)
            action = Bernoulli(prob).sample().item()
        if action == bool(1): # not stop
            g.add_nodes(1)
            self.initialize_node(g, graph_embed)

        log_prob = bernoulli_action_log_prob(logit, action)

        # print("lp is", action, log_prob)
        self.log_prob.append(log_prob)

        return bool(action), log_prob

class AddEdges(nn.Module):
    def __init__(self, graph_embed_func, node_dim, num_edge_types):
        '''
        graph_embed_func: module of type GraphEmbed above
        node_dim: node embedding dimension
        num_edge_types: length of one hot vector representing edge
        '''
        super(AddEdges, self).__init__()

        self.num_edge_types = num_edge_types
        self.graph_embed_func = graph_embed_func
        self.add_edge = nn.Sequential(
            nn.Linear(graph_embed_func.graph_dim + 2 * node_dim, graph_embed_func.graph_dim + 2 * node_dim),
            nn.ReLU(True),
            nn.Linear(graph_embed_func.graph_dim + 2 * node_dim, num_edge_types+1)
        ) # index 0 being no edge
        self.log_prob = []

        # Static edge types

    def forward(self, g, actions=None):
        '''
        Actions needs to be list of length |current number of nodes - 1|
        with value 0 if no edge is added and |edge type idx + 1| if an edge is added
        '''
        N = g.number_of_nodes()
        graph_embed = self.graph_embed_func(g).repeat(N-1, 1)
        curr_embed = g.nodes[N-1].data['hv'].repeat(N-1, 1) # 1, node_dim
        prev_embed = g.ndata['hv'][:N-1] # N-1, node_dim


        logits = self.add_edge(torch.cat(
            [graph_embed, prev_embed, curr_embed], dim=1 # Check dims
        )) # N-1, edge_types + 1

        #node_probs = torch.sigmoid(logits, dim=0)
        if actions == None:
            # print("logits shape is", logits.shape)
            # print(torch.softmax(logits, dim=-1))
            # print(Categorical(logits=logits).sample())
            actions = Categorical(logits=logits).sample().numpy().tolist()
            # print("Generated actions", actions)
        for idx, etype in enumerate(actions):
            if etype == 0: continue
            e_em = torch.zeros(1, self.num_edge_types)
            e_em[0,etype-1] = 1.0

            g.add_edges(N-1, idx)
            g.add_edges(idx, N-1)
            # print('adding edge between', idx, N-1)
            g.edges[N-1, idx].data['he'] = e_em
            # print(idx, etype, e_em)

            g.edges[idx, N-1].data['he'] = e_em
            # print(g.edata)

        log_prob = F.log_softmax(logits, dim=1).gather(1, torch.tensor(actions).long().view(-1, 1)).sum()
        self.log_prob.append(log_prob)
        return actions, log_prob

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
