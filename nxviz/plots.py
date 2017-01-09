from .geometry import node_theta, get_cartesian, circos_radius, group_theta
from .utils import (infer_data_type, num_discrete_groups, cmaps,
                    is_data_diverging)
from matplotlib.path import Path
from matplotlib.cm import get_cmap
from collections import defaultdict
from pprint import pprint

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def despine(ax):
    for spine in ax.spines:
        ax.spines[spine].set_visible(False)
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)


class BasePlot(object):
    """
    BasePlot: An extensible class for designing new network visualizations.

    The BasePlot constructor takes in a NetworkX graph object, and a series of
    keyword arguments specifying how nodes and edges should be styled and
    ordered.

    An optional data_types dictionary can be passed in to bypass data type
    inference.
    """
    def __init__(self, graph, node_order=None, node_size=None,
                 node_grouping=None, node_color=None, edge_width=None,
                 edge_color=None, data_types=None, nodeprops=None,
                 edgeprops=None):
        super(BasePlot, self).__init__()
        # Set graph object
        self.graph = graph
        self.nodes = graph.nodes()  # keep track of nodes separately.

        # Set node arrangement
        self.node_order = node_order
        self.node_grouping = node_grouping
        self.group_and_sort_nodes()

        # Set node radius
        self.node_size = node_size

        # Set node colors
        self.node_color = node_color
        if self.node_color:
            self.node_colors = []
            self.compute_node_colors()
        else:
            self.node_colors = ['blue'] * len(self.nodes)

        # Set edge properties
        self.edge_width = edge_width
        self.edge_color = edge_color

        # Set data_types dictionary
        if not data_types:
            self.data_types = dict()
        else:
            self.check_data_types(data_types)
            self.data_types = data_types

        self.figure = plt.figure(figsize=(6, 6))
        self.ax = self.figure.add_subplot(1, 1, 1)
        # despine(self.ax)

        # We provide the following attributes that can be set by the end-user.
        # nodeprops are matplotlib patches properties.
        if nodeprops:
            self.nodeprops = nodeprops
        else:
            self.nodeprops = {'radius': 1}
        # edgeprops are matplotlib line properties. These can be set after
        # instantiation but before calling the draw() function.
        if edgeprops:
            self.edgeprops = edgeprops
        else:
            self.edgeprops = {'facecolor': 'none',
                              'alpha': 0.2}

    def check_data_types(self, data_types):
        """
        Checks the data_types passed into the Plot constructor and makes sure
        that:
        - the values passed in belong to 'ordinal', 'categorical', or
          'continuous'.
        """
        for k, v in data_types.items():
            assert v in ['ordinal', 'categorical', 'continuous']

    def draw(self):
        self.draw_nodes()
        self.draw_edges()
        self.ax.relim()
        self.ax.autoscale_view()

    def compute_node_colors(self):
        """
        Computes the node colors.
        """
        data = [self.graph.node[n][self.node_color] for n in self.nodes]
        data_reduced = sorted(list(set(data)))
        dtype = infer_data_type(data)
        n_grps = num_discrete_groups(data)

        if dtype == 'categorical' or dtype == 'ordinal':
            cmap = get_cmap(cmaps['Accent_{0}'.format(n_grps)].mpl_colormap)
        elif dtype == 'continuous' and not is_data_diverging(data):
            cmap = get_cmap(cmaps['continuous'].mpl_colormap)
        elif dtype == 'continuous' and is_data_diverging(data):
            cmap = get_cmap(cmaps['diverging'].mpl_colormap)

        for d in data:
            idx = data_reduced.index(d) / n_grps
            self.node_colors.append(cmap(idx))

    def compute_node_positions(self):
        """
        Computes the positions of each node on the plot.

        Needs to be implemented for each plot type.
        """
        pass

    def draw_nodes(self):
        """
        Renders the nodes to the plot or screen.

        Needs to be implemented for each plot type.
        """
        pass

    def draw_edges(self):
        """
        Renders the nodes to the plot or screen.

        Needs to be implemented for each plot type.
        """
        pass

    def group_and_sort_nodes(self):
        """
        Groups and then sorts the nodes according to the criteria passed into
        the Plot constructor.
        """
        if self.node_grouping and not self.node_order:
            self.nodes = [n for n, d in
                          sorted(self.graph.nodes(data=True),
                                 key=lambda x: x[1][self.node_grouping])]

        elif self.node_order and not self.node_grouping:
            self.nodes = [n for n, _ in
                          sorted(self.graph.nodes(data=True),
                                 key=lambda x: x[1][self.node_order])]

        elif self.node_grouping and self.node_order:
            self.nodes = [n for n, d in
                          sorted(self.graph.nodes(data=True),
                                 key=lambda x: (x[1][self.node_grouping],
                                                x[1][self.node_order]))]


class CircosPlot(BasePlot):
    """
    Plotting object for CircosPlot.
    """
    def __init__(self, graph, node_order=None, node_size=None,
                 node_grouping=None, node_color=None, edge_width=None,
                 edge_color=None, data_types=None, nodeprops=None,
                 edgeprops=None):

        # Initialize using BasePlot
        BasePlot.__init__(self, graph, node_order=node_order,
                          node_size=node_size, node_grouping=node_grouping,
                          node_color=node_color, edge_width=edge_width,
                          edge_color=edge_color, data_types=data_types,
                          nodeprops=nodeprops, edgeprops=edgeprops)
        # Compute each node's positions.
        self.compute_node_positions()

    def compute_node_positions(self):
        """
        Uses the get_cartesian function to compute the positions of each node
        in the Circos plot.
        """
        xs = []
        ys = []
        node_r = self.nodeprops['radius']
        radius = circos_radius(n_nodes=len(self.graph.nodes()), node_r=node_r)
        self.plot_radius = radius
        self.nodeprops['linewidth'] = radius * 0.01
        for node in self.nodes:
            x, y = get_cartesian(r=radius, theta=node_theta(self.nodes, node))
            xs.append(x)
            ys.append(y)
        self.node_coords = {'x': xs, 'y': ys}

    def draw_nodes(self):
        """
        Renders nodes to the figure.
        """
        node_r = self.nodeprops['radius']
        lw = self.nodeprops['linewidth']
        for i, node in enumerate(self.nodes):
            x = self.node_coords['x'][i]
            y = self.node_coords['y'][i]
            color = self.node_colors[i]
            node_patch = patches.Circle((x, y), node_r,
                                        lw=lw, color=color,
                                        zorder=2)
            self.ax.add_patch(node_patch)

    def draw_edges(self):
        """
        Renders edges to the figure.
        """
        for i, (start, end) in enumerate(self.graph.edges()):
            start_theta = node_theta(self.nodes, start)
            end_theta = node_theta(self.nodes, end)
            verts = [get_cartesian(self.plot_radius, start_theta),
                     (0, 0),
                     get_cartesian(self.plot_radius, end_theta)]
            codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]

            path = Path(verts, codes)
            patch = patches.PathPatch(path, lw=1, **self.edgeprops, zorder=1)
            self.ax.add_patch(patch)


class HivePlot(BasePlot):
    """
    Plotting object for HivePlot. With HivePlots, node_grouping and node_groups
    must be specified; these are non-optional.
    """
    def __init__(self, graph, node_grouping, node_groups,
                 node_order=None, node_size=None, node_color=None,
                 edge_width=None, edge_color=None, data_types=None,
                 nodeprops=None, edgeprops=None):

        # Defensive checks prior to initialization.

        # 1. Check that there are either 2 or 3 node_groups specified.
        assert len(node_groups) in [2, 3]
        # 2. Should check that node_grouping is specified, but because it's not
        #    a keyword argument, it must be specified.
        # 3. Should check that node_grouping exists as a key in the node data,
        #    but because in the initialization we call on
        #    self.group_and_sort_nodes(),we will definitely get an error if the
        #    key is not present. Can change this assumption in the future.
        # 4. Check that node_groups is a list of elements.
        assert isinstance(node_groups, list), "node_groups should be a list"
        # 5. Check that every node has the node_grouping key in its attributes
        #    dictionary.
        for n, d in graph.nodes(data=True):
            assert node_grouping in d.keys(), "{0} not in {1}".format(node_grouping, d.keys())
        # 6. Check that node_grouping is not None.
        assert node_grouping is not None, "node_grouping cannot be None"
        # 7. Check that the group_vals is a subset of all possible values keyed
        #    by the node_grouping key.
        group_vals = set([d[node_grouping] for n, d in graph.nodes(data=True)])
        assert set(node_groups).issubset(group_vals), '{0} not a subset of {1}'.format(node_groups, group_vals)

        # Initialize using BasePlot
        BasePlot.__init__(self, graph, node_order=node_order,
                          node_size=node_size, node_grouping=node_grouping,
                          node_color=node_color, edge_width=edge_width,
                          edge_color=edge_color, data_types=data_types,
                          nodeprops=nodeprops, edgeprops=edgeprops)

        # Set some hiveplot-specific attributes.
        # Firstly, create a dictionary of lists to contain only the nodes that
        # are to be plotted.
        self.node_groups = node_groups
        self.grouped_nodes = defaultdict(list)
        for n in self.nodes:
            key = self.node_grouping
            value = self.graph.node[n][key]
            if value in self.node_groups:
                self.grouped_nodes[value].append(n)

        # Next up, we create a subgraph that houses only the nodes and edges
        # to be plotted.
        subG_nodes = []
        for nodes in self.grouped_nodes.values():
            subG_nodes.extend(nodes)
        self.subgraph = self.graph.subgraph(subG_nodes)

        # Next up, decide whether we want to use a split axis or not.
        # This is decided by checking whether there are edges within groups.
        self.split_axis = self.has_edge_within_groups()
        if self.split_axis:
            self.minor_angle = np.pi / 12

        # Next up, we set the internal radius.
        self.internal_radius = 2

        # Compute the node thetas upfront. This is distinct from the node
        # node positions, which are (x, y) coordinates. We will use the
        # node_thetas information to compute the (x, y) coordinates.
        self.thetas = defaultdict(dict)
        self.compute_node_thetas()

        # Compute each node's positions.
        self.node_coords = dict()
        self.compute_node_positions()
        # pprint(self.node_coords)

    def compute_node_thetas(self):
        """
        Computes the thetas for each node. As nodes are a continguous list that
        are iterable (and thus index-able), we keep the node_thetas in the same
        format.
        """
        for group, nodes in self.grouped_nodes.items():
            major_angle = group_theta(self.node_groups, group)
            for node in nodes:
                if self.split_axis:
                    minus_angle = major_angle - self.minor_angle
                    plus_angle = major_angle + self.minor_angle

                    self.thetas[group]['minus'] = minus_angle
                    self.thetas[group]['plus'] = plus_angle
                else:
                    self.thetas[group]['axis'] = major_angle

    def has_edge_within_groups(self):
        """
        Checks whether there are within-group edges or not across the nodes
        that are being plotted.
        """
        result = False
        for group, nodes in self.grouped_nodes.items():
            sG = self.graph.subgraph(nodes)
            if len(sG.edges()) > 0:
                result = True
                break

        return result

    def compute_node_positions(self):
        """
        Computes the positions of each node on the plot.

        Sets the node_coords attribute inherited from BasePlot.

        Uses self.thetas
        """
        xs = defaultdict(lambda: defaultdict(list))
        ys = defaultdict(lambda: defaultdict(list))

        # pprint(self.thetas)

        for group, nodes in self.grouped_nodes.items():
            for r, node in enumerate(nodes):
                if self.split_axis:
                    theta = self.thetas[group]['minus']
                    x, y = get_cartesian(r+2, theta)
                    xs[group]['minus'].append(x)
                    ys[group]['minus'].append(y)

                    theta = self.thetas[group]['plus']
                    x, y = get_cartesian(r+2, theta)
                    xs[group]['plus'].append(x)
                    ys[group]['plus'].append(y)
                else:
                    theta = self.thetas[group]['axis']
                    x, y = get_cartesian(r+2, theta)
                    xs[group]['axis'].append(x)
                    ys[group]['axis'].append(y)
        self.node_coords['x'] = xs
        self.node_coords['y'] = ys

    def draw_nodes(self):
        """
        Renders nodes to the figure.
        """
        for group, nodes in self.grouped_nodes.items():
            for r, n in enumerate(nodes):
                if self.split_axis:
                    x = self.node_coords['x'][group]['minus'][r]
                    y = self.node_coords['y'][group]['minus'][r]
                    self.draw_node(n, x, y)

                    x = self.node_coords['x'][group]['plus'][r]
                    y = self.node_coords['y'][group]['plus'][r]
                    self.draw_node(n, x, y)

                else:
                    x = self.node_coords['x'][group]['axis'][r]
                    y = self.node_coords['y'][group]['axis'][r]
                    self.draw_node(n, x, y)

    def draw_node(self, n, x, y):
        """
        Convenience function for simplifying the code in draw_nodes().
        """
        node_idx = self.nodes.index(n)
        circle = plt.Circle(xy=(x, y), radius=0.2,
                            color=self.node_colors[node_idx], zorder=2)
        self.ax.add_patch(circle)

    def draw_edges(self):
        """
        Renders edges to the figure.
        """
        G = self.subgraph
        for n1, n2 in G.edges():
            self.draw_edge(n1, n2)

    def draw_edge(self, n1, n2):
        """
        Convenience function for plotting edges.
        """
        start_grp = self.graph.node[n1][self.node_grouping]
        end_grp = self.graph.node[n2][self.node_grouping]

        start_theta = group_theta(self.node_groups, start_grp)
        end_theta = group_theta(self.node_groups, end_grp)

        pprint(start_theta)
        pprint(end_theta)

        start_theta, end_theta = self.correct_thetas(start_theta, end_theta)
        mid_theta = (start_theta + end_theta) / 2

        start_idx = self.grouped_nodes[start_grp].index(n1)
        end_idx = self.grouped_nodes[end_grp].index(n2)

        start_radius = start_idx + self.internal_radius
        end_radius = end_idx + self.internal_radius

        middle1_radius = np.min([start_radius, end_radius])
        middle2_radius = np.max([start_radius, end_radius])

        if start_radius > end_radius:
            middle1_radius, middle2_radius = middle2_radius, middle1_radius

        startx, starty = get_cartesian(start_radius, start_theta)
        mid1x, mid1y = get_cartesian(middle1_radius, mid_theta)
        mid2x, mid2y = get_cartesian(middle2_radius, mid_theta)
        endx, endy = get_cartesian(end_radius, end_theta)

        verts = [(startx, starty),
                 (mid1x, mid1y),
                 (mid2x, mid2y),
                 (endx, endy),
                 ]

        pprint(verts)

        codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]

        path = Path(verts, codes)
        patch = patches.PathPatch(path, facecolor='none',
                                  alpha=0.3, zorder=1)
        self.ax.add_patch(patch)

    def correct_thetas(self, start_theta, end_theta):
        """
        Corrects the start and end angles.
        """
        # Edges going the anti-clockwise direction involves angle = 0.
        if start_theta == 0 and (end_theta - start_theta > np.pi):
            start_theta = np.pi * 2
        if end_theta == 0 and (end_theta - start_theta < -np.pi):
            end_theta = np.pi * 2

        # Case when we split axis:
        # if self.split_axis:
        if start_theta == end_theta:
            start_theta = start_theta - self.minor_angle
            end_theta = end_theta + self.minor_angle
            # elif start_theta > end_theta:
            #     start_theta = start_theta - self.minor_angle
            #     end_theta = end_theta + self.minor_angle

        # Case when start_theta is greater than end_theta.
        if start_theta > end_theta and end_theta == -np.pi:
            end_theta = np.pi
        return start_theta, end_theta



    def draw(self):
        self.draw_nodes()
        self.draw_edges()
        self.ax.relim()
        self.ax.autoscale_view()


class MatrixPlot(BasePlot):
    """
    Plotting object for the MatrixPlot.
    """
    def __init__(self, graph, node_order=None, node_size=None,
                 node_grouping=None, node_color=None, edge_width=None,
                 edge_color=None, data_types=None, nodeprops=None,
                 edgeprops=None):

        # Initialize using BasePlot
        BasePlot.__init__(self, graph, node_order=node_order,
                          node_size=node_size, node_grouping=node_grouping,
                          node_color=node_color, edge_width=edge_width,
                          edge_color=edge_color, data_types=data_types,
                          nodeprops=nodeprops, edgeprops=edgeprops)

        # The following atribute is specific to MatrixPlots
        self.cmap = cmaps['continuous'].mpl_colormap

    def draw(self):
        """
        Draws the plot to screen.
        """
        matrix = nx.to_numpy_matrix(self.graph, nodelist=self.nodes)
        self.ax.matshow(matrix, cmap=self.cmap)


class ArcPlot(BasePlot):
    """
    Plotting object for ArcPlot.
    """
    def __init__(self, graph, node_order=None, node_size=None,
                 node_grouping=None, node_color=None, edge_width=None,
                 edge_color=None, data_types=None, nodeprops=None,
                 edgeprops=None):

        # Initialize using BasePlot
        BasePlot.__init__(self, graph, node_order=node_order,
                          node_size=node_size, node_grouping=node_grouping,
                          node_color=node_color, edge_width=edge_width,
                          edge_color=edge_color, data_types=data_types,
                          nodeprops=nodeprops, edgeprops=edgeprops)
        # Compute each node's positions.
        self.compute_node_positions()

    def compute_node_positions(self):
        """
        Computes nodes positions.

        Arranges nodes in a line starting at (x,y) = (0,0). Node radius is
        assumed to be equal to 0.5 units. Nodes are placed at integer
        locations.
        """
        xs = []
        ys = []

        for node in self.nodes:
            xs.append(self.nodes.index(node))
            ys.append(0)

        self.node_coords = {'x': xs, 'y': ys}

    def draw_nodes(self):
        """
        Draw nodes to screen.
        """
        node_r = 1
        for i, node in enumerate(self.nodes):
            x = self.node_coords['x'][i]
            y = self.node_coords['y'][i]
            color = self.node_colors[i]
            node_patch = patches.Ellipse((x, y), node_r, node_r,
                                         lw=0, color=color, zorder=2)
            self.ax.add_patch(node_patch)

    def draw_edges(self):
        """
        Renders edges to the figure.
        """
        for i, (start, end) in enumerate(self.graph.edges()):
            start_idx = self.nodes.index(start)
            start_x = self.node_coords['x'][start_idx]
            start_y = self.node_coords['y'][start_idx]

            end_idx = self.nodes.index(end)
            end_x = self.node_coords['x'][end_idx]
            end_y = self.node_coords['y'][end_idx]

            arc_radius = abs(end_x - start_x) / 2
            # we do min(start_x, end_x) just in case start_x is greater than
            # end_x.
            middle_x = min(start_x, end_x) + arc_radius
            middle_y = arc_radius * 2

            verts = [(start_x, start_y),
                     (middle_x, middle_y),
                     (end_x, end_y)]

            codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]

            path = Path(verts, codes)
            patch = patches.PathPatch(path, lw=1, **self.edgeprops, zorder=1)
            self.ax.add_patch(patch)

    def draw(self):
        self.draw_nodes()
        self.draw_edges()
        xlimits = (-1, len(self.nodes) + 1)
        # halfwidth = len(self.nodes) + 1 / 2
        # ylimits = (-halfwidth, halfwidth)
        self.ax.set_xlim(*xlimits)
        self.ax.set_ylim(*xlimits)
