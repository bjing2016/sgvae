from models import *
import torch

# Limitations to note: one-hot edge and node labels
#       All neural networls are only one layer
#       Undirected (bidirectional) edges
#       Edge types are initialized one hot and static

if __name__ == "__main__":
    g = dgl.DGLGraph()
    # add 34 nodes into the graph; nodes are labeled from 0~33
    
    g.add_nodes(1)
    '''
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
    '''
    #for (src, dest) in edge_list:
    #    g.edges[src, dest].data['he'] = torch.ones(1,1)
    
    g.ndata['hv'] = torch.randn(1, 5)   

    graph_embed_func = GraphEmbed(5, 10)
    print(graph_embed_func(g))

    #def send_source(edges): return {'m': edges.src['hv'] + 1}
    #g.register_message_func(send_source)
    #g.send(g.edges())

    for i in range(100):
        graph_prop = GraphProp(3, 5, 5, 1)
        graph_prop(g)

        node_adder = AddNode(graph_embed_func, 5, 5)
        added, prob = node_adder(g, action=True)
        print('node',prob)
        #if added:
        edge_adder = AddEdges(graph_embed_func, 5, 1)
        print(edge_adder(g)[1])


    print('compiled without issue')
