import torch.nn as nn
import dgl
from util import *
class GraphDestructor(nn.Module):
    # returns the inverse order of nodes and edges by destruction order
    NotImplemented

'''
Implementation adapted from https://docs.dgl.ai/en/latest/tutorials/models/3_generative_model/5_dgmg.html
'''
class GraphConstructor(nn.Module):
    # TODO: 
    def __init__(self, z_dim, dims):
        super(GraphConstructor, self).__init__()

    def construct(self):
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
        return {'m' : torch.cat([edges.src['hv'], edges.dest['hv'],
                                    edges.data['he']], dim = 1)}

    def dgmg_reduce(self, nodes, round):
        hv_old = nodes.data['hv']
        m = nodes.mailbox['m']
        message = torch.cat([
            hv_old.unsqueeze(1).expand(-1, m.size(1), -1), m], dim = 2)
        node_activation = (self.message_funcs[round](message)).sum(1)

        return {'a':node_activation}
    
    def forward(self, g):
        if g.number_of_edges() > 0:
            for t in range(self.rounds):
                g.update_all(message_func=self.gdmg_msg, #the message func field is just the gathering step
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

        self.init_node_activation = torch.zeroes(1, node_act_dim) # To satisfy the GRU cell

        self.log_prob = []

    def initialize_node(self, g, graph_embed):
        hv_init = self.node_init(graph_embed)
        g.nodes[-1].data['hv'] = hv_init
        g.nodes[-1].data['a'] = self.init_node_activation

    def forward(self, g, action=None):
        graph_embed = self.graph_embed_func(g)
        logit = self.add_node(graph_embed)
        prob = torch.sigmoid(logit)

        if action == None:
            action = Bernoulli(prob).sample().item()
        if acction == 1: # not stop
            g.add_nodes(1)
            self.initialize_node(g, graph_embed)
        
        log_prob = bernoulli_action_log_prob(logit, action)
        self.log_prob.append(log_prob)
        
        return action

class AddEdge(nn.Module):
    def __init__(self, graph_embed_func, node_dim, num_edge_types):
        super(AddEdge, self).__init__()

        self.graph_embed_func = graph_embed_func
        self.add_edge = nn.Linear(graph_embed_func.graph_dim + node_dim, num_edge_types)
        
        self.log_prob = []

        # Static edge types

    def forward(self, g, action=None):
        graph_embed = self.graph_embed_func(g)

        src_embed = g.nodes[-1].data['hv']

        logits = self.self.add_edge(torch.cat(
            [graph_embed, src_embed], dim=1
        ))
        # Incomplete


        
        



if __name__ == "__main__":
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
