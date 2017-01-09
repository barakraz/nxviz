import networkx as nx
import matplotlib.pyplot as plt
from nxviz.plots import HivePlot

nodes = range(1, 8)
edges = [(1, 2),
         (2, 3),
         (1, 4)]

G = nx.Graph()
G.add_nodes_from(nodes)
# G.add_edges_from(edges)
# G.add_edge(1, 2)
G.add_edge(2, 3)
G.add_edge(6, 3)

for n in G.nodes():
    G.node[n]['class'] = n % 3

h = HivePlot(graph=G, node_grouping='class',
             node_groups=[0, 1, 2], node_color='class')
h.draw()
plt.show()
