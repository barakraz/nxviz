"""
Displays a NetworkX barbell graph to screen using a HivePlot.
"""

from nxviz.plots import HivePlot
import networkx as nx
import matplotlib.pyplot as plt

G = nx.barbell_graph(m1=10, m2=3)
groups = ['one', 'two', 'three', 'four']
for n, d in G.nodes(data=True):
    G.node[n]['class'] = groups[n % 4]
    print(G.node[n])
node_groups = ['one', 'three', 'four']
c = HivePlot(G, node_grouping="class", node_groups=node_groups,
             node_color="class", node_order='class')
c.draw()
plt.show()
