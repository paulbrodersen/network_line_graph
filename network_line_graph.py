#!/usr/bin/env python
# -*- coding: utf-8 -*-

# network_line_graph: visualisation to compare different states of a network.

# Copyright (C) 2016 Paul Brodersen <paulbrodersen+network_line_graph@gmail.com>

# Author: Paul Brodersen <paulbrodersen+network_line_graph@gmail.com>

# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

"""
Network Line Graph
==================

Visualise a network by assigning nodes a position along the
x-axis and then drawing the edges between them as Bezier curves.

Such a visualisation is particularly useful to compare two different
network states, as a second network state can be drawn in the same way
below the x-axis. The symmetry around the x-axis accentuates changes
in the network structure.

Example:
--------

import numpy as np
import matplotlib.pyplot as plt
import network_line_graph as nlg

# initialise figure
fig, ax = plt.subplots(1,1)

# make a weighted random graph
n = 20 # number of nodes
p = 0.1 # connection probability
a1 = np.random.rand(n,n) < p # adjacency matrix
w1 = np.random.randn(n,n) # weight matrix
w1[~a1] = np.nan

# plot connections above x-axis
nlg.draw(w1, arc_above=True, ax=ax)

# make another weighted random graph;
a2 = np.random.rand(n,n) < p # adjacency matrix
w2 = np.random.randn(n,n) # weight matrix
w2[~a2] = np.nan

# plot connections below x-axis
nlg.draw(w2, arc_above=False, ax=ax)

# annotate
ax.text(0,1, 'Graph 1', transform=ax.transAxes, fontsize=18)
ax.text(0,0, 'Graph 2', transform=ax.transAxes, fontsize=18)

plt.show()

"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.cbook as cb
from matplotlib.colors import colorConverter, Colormap
from matplotlib.patches import FancyArrowPatch, Circle

def draw(adjacency_matrix, node_order=None, node_labels=None, ax=None, **kwargs):
    """
    Convenience function that tries to do "the right thing".

    For a full list of available arguments, and
    for finer control of the individual draw elements,
    please refer to the documentation of

        draw_nodes()
        draw_edges()
        draw_node_labels()
        draw_edge_labels()

    Arguments
    ----------
    adjacency_matrix: (n, n) numpy.ndarray
        Adjacency or weight matrix of the network.

    node_order : (n, ) numpy.ndarray
        Order in which the nodes are plotted along the x-axis.
        If unspecified and networkx is installed, the node order is
        set such that nodes, which are strongly connected with each
        other, occur close in the node order.

    ax : matplotlib.axis instance or None (default None)
       Axis to plot onto; if none specified, one will be instantiated with plt.gca().

    Examples
    --------

    See Also
    --------
    draw_nodes()
    draw_edges()
    draw_node_labels()
    draw_edge_labels()

    """

    if ax is None:
        ax = plt.gca()

    if not np.all(adjacency_matrix == adjacency_matrix.T): # i.e. directed
        kwargs.setdefault('draw_arrows', True)

    if len(np.unique(adjacency_matrix)) > 2: # i.e. more than 0s and 1s i.e. weighted

        # reorder edges such that edges with large absolute weights are plotted last
        # and hence most prominent in the graph
        weights = adjacency_matrix.copy()
        edge_zorder = np.abs(weights) / np.float(np.nanmax(np.abs(weights)))
        edge_zorder *= np.sum(~np.isnan(weights))

        # apply edge_vmin, edge_vmax
        edge_vmin = kwargs.get('edge_vmin', np.nanmin(weights))
        edge_vmax = kwargs.get('edge_vmax', np.nanmax(weights))
        weights[weights<edge_vmin] = edge_vmin
        weights[weights>edge_vmax] = edge_vmax

        # rescale weights such that
        #  - the colormap midpoint is at zero-weight, and
        #  - negative and positive weights have comparable intensity values
        weights /= np.nanmax([np.nanmax(abs(weights)), np.abs(edge_vmax), np.abs(edge_vmin)]) # [-1, 1]
        weights += 1. # [0, 2]
        weights /= 2. # [0, 1]

        kwargs.setdefault('edge_color', weights)
        # kwargs.setdefault('edge_vmin', 0.)
        # kwargs.setdefault('edge_vmax', 1.)
        kwargs['edge_vmin'] = 0.
        kwargs['edge_vmax'] = 1.
        kwargs.setdefault('edge_cmap', 'RdGy')
        kwargs.setdefault('edge_zorder', edge_zorder)

    number_of_nodes = adjacency_matrix.shape[0]

    if node_order is None:
        try:
            node_order = _optimize_node_order(adjacency_matrix)
        except:
            node_order = np.arange(number_of_nodes)

    node_positions = _get_positions(node_order)

    node_artists = draw_nodes(node_positions, **kwargs)
    edge_artists = draw_edges(adjacency_matrix, node_positions, node_artists, **kwargs)

    if node_labels is not None:
        draw_node_labels(node_positions, node_labels)

    # Patches are not registered properly
    # when matplotlib sets axis limits automatically.
    # So we need to do this manually.
    _update_view(adjacency_matrix, node_positions, ax)

    # remove superfluous ink
    _make_pretty(ax)

    return

def _get_positions(node_order):
    return np.c_[node_order, np.zeros((len(node_order)))]

def _optimize_node_order(adjacency_matrix):
    """
    Improve node order by grouping strongly connected nodes closer to each other.
    Essentially, the graph is recursively partitioned using minimum flow cuts and
    the order of the nodes in the resulting hierarchical clustering is returned.
    """

    import networkx

    # make networkx compatible
    w = adjacency_matrix.copy()
    w[np.isnan(w)] = 0.

    # graph cuts only implemented for undirected graphs with positive weights
    # in networkx
    w = np.abs(w)
    w = w + w.T

    g = networkx.from_numpy_matrix(w)
    partitions = [range(len(w))]
    while np.max([len(p) for p in partitions]) > 2:
        new_partitions = []
        for ii, p0 in enumerate(partitions):
            if len(p0) > 2:
                c, (p1, p2) = networkx.stoer_wagner(g.subgraph(p0))
                new_partitions.append(p1)
                new_partitions.append(p2)
            else: # nothing to partition
                new_partitions.append(p0)
        partitions = new_partitions

    node_order = np.concatenate(partitions)
    return node_order

def draw_nodes(node_positions,
               node_shape='full',
               node_size=20.,
               node_edge_width=4.,
               node_color='w',
               node_edge_color='k',
               cmap=None,
               vmin=None,
               vmax=None,
               node_alpha=1.0,
               ax=None,
               **kwds):
    """
    Draw node markers at specified positions.

    Arguments
    ----------
    node_positions : (n, 2) numpy.ndarray
        iterable of (x,y) node positions

    node_shape : string (default 'full')
       The shape of the node. One of 'full', 'top half', 'bottom half'.

    node_size : scalar or (n,) numpy array (default 3.)
       Size (radius) of nodes.
       A node size of 1 corresponds to a length of 0.01 in node position units.

    node_edge_width : [scalar | sequence] (default 0.5)
       Line width of node marker border.

    node_color : color string, or array of floats (default 'w')
       Node color. Can be a single color format string
       or a sequence of colors with the same length as node_positions.
       If numeric values are specified they will be mapped to
       colors using the cmap and vmin/vmax parameters.

    node_edge_color : color string, or array of floats (default 'k')
       Node color. Can be a single color format string,
       or a sequence of colors with the same length as node_positions.
       If numeric values are specified they will be mapped to
       colors using the cmap and vmin,vmax parameters.

    cmap : Matplotlib colormap (default None)
       Colormap for mapping intensities of nodes.

    vmin, vmax : floats (default None)
       Minimum and maximum for node colormap scaling.

    alpha : float (default 1.)
       The node transparency.

    ax : Matplotlib Axes object, optional
       Draw the graph in the specified Matplotlib axes.

    Returns
    -------
    artists: dict
        Dictionary mapping node index to the node face artist and node edge artist,
        where both artists are instances of matplotlib.patches.
        Node face artists are indexed with keys of the format (index, 'face'),
        Node edge artists are indexed with keys (index, 'edge').

    """

    if ax is None:
        ax = plt.gca()

    # convert all node properties that not iterable into iterable formats
    number_of_nodes = len(node_positions)
    node_color = _parse_color_input(number_of_nodes, node_color, cmap, vmin, vmax, node_alpha)
    node_edge_color = _parse_color_input(number_of_nodes, node_edge_color, cmap, vmin, vmax, node_alpha)

    if isinstance(node_size, (int, float)):
        node_size = node_size * np.ones((number_of_nodes))
    if isinstance(node_edge_width, (int, float)):
        node_edge_width = node_edge_width * np.ones((number_of_nodes))

    # rescale
    node_size = node_size.astype(np.float) * 1e-2
    node_edge_width = node_edge_width.astype(np.float) * 1e-2

    # circles made with plt.scatter scale with axis dimensions
    # which in practice makes it hard to have one consistent layout
    # -> use patches.Circle instead which creates circles that are in data coordinates
    artists = dict()
    for ii in range(number_of_nodes):
        # simulate node edge by drawing a slightly larger node artist;
        # I wish there was a better way to do this,
        # but this seems to be the only way to guarantee constant proportions,
        # as linewidth argument in matplotlib.patches will not be proportional
        # to radius as it is in axis coordinates
        node_edge_artist = _get_node_artist(shape=node_shape,
                                            position=node_positions[ii],
                                            size=node_size[ii],
                                            facecolor=node_edge_color[ii],
                                            zorder=2)
        ax.add_artist(node_edge_artist)
        artists[(ii, 'edge')] = node_edge_artist

        # draw node
        node_artist = _get_node_artist(shape=node_shape,
                                       position=node_positions[ii],
                                       size=node_size[ii] -node_edge_width[ii],
                                       facecolor=node_color[ii],
                                       zorder=2)
        ax.add_artist(node_artist)
        artists[(ii, 'face')] = node_artist

    return artists

def _get_node_artist(shape, position, size, facecolor, zorder=2):
    if shape == 'full': # full circle
        artist = matplotlib.patches.Circle(xy=position,
                                           radius=size,
                                           facecolor=facecolor,
                                           linewidth=0.,
                                           zorder=zorder)
    elif shape == 'top half':
        NotImplementedError
    elif shape == 'bottom half':
        NotImplementedError
    else:
        raise ValueError("Node shape one of: 'full'. Current shape:{}".format(shape))

    return artist

# verbatim in netgraph
def draw_node_labels(node_positions,
                     node_labels,
                     font_size=8,
                     font_color='k',
                     font_family='sans-serif',
                     font_weight='normal',
                     font_alpha=1.,
                     bbox=None,
                     clip_on=False,
                     ax=None,
                     **kwargs):
    """
    Draw node labels.

    Arguments
    ---------
    node_positions : (n, 2) numpy.ndarray
        (x, y) node coordinates.

    node_labels : dict
       Dictionary mapping node indices to labels.
       Only nodes in the dictionary are labelled.

    font_size : int (default 12)
       Font size for text labels

    font_color : string (default 'k')
       Font color string

    font_family : string (default='sans-serif')
       Font family

    font_weight : string (default='normal')
       Font weight

    font_alpha : float (default 1.)
       Text transparency

    bbox : Matplotlib bbox
       Specify text box shape and colors.

    clip_on : bool
       Turn on clipping at axis boundaries (default=False)

    ax : matplotlib.axis instance or None (default None)
       Draw the graph in the specified Matplotlib axis.

    Returns
    -------
    artists: dict
        Dictionary mapping node indices to text objects.

    @reference
    Borrowed with minor modifications from networkx/drawing/nx_pylab.py

    """

    if ax is None:
        ax = plt.gca()

    # set optional alignment
    horizontalalignment = kwargs.get('horizontalalignment', 'center')
    verticalalignment = kwargs.get('verticalalignment', 'center')

    artists = dict()  # there is no text collection so we'll fake one
    for ii, label in node_labels.iteritems():
        x, y = node_positions[ii]
        text_object = ax.text(x, y,
                              label,
                              size=font_size,
                              color=font_color,
                              alpha=font_alpha,
                              family=font_family,
                              weight=font_weight,
                              horizontalalignment=horizontalalignment,
                              verticalalignment=verticalalignment,
                              transform=ax.transData,
                              bbox=bbox,
                              clip_on=False)
        artists[ii] = text_object

    return artists

def draw_edges(adjacency_matrix,
               node_positions,
               node_artists=None,
               edge_width=2.,
               edge_color='k',
               edge_cmap=None,
               edge_vmin=None,
               edge_vmax=None,
               edge_alpha=1.,
               edge_zorder=None,
               ax=None,
               arc_above=True,
               draw_arrows=True,
               **arrow_kwargs):
    """

    Draw the edges of the network.

    Arguments
    ----------
    adjacency_matrix: (n, n) numpy.ndarray
        Adjacency or weight matrix of the network.

    node_positions : (n, 2) numpy.ndarray
        (x, y) node coordinates

    node_artists: dictionary
        Container of node_artists as returned by draw_nodes() or None (default None).
        If node artists are provided, edges start and end at the edge of the node artist,
        not at the node positions (i.e. their centre).

    edge_width : float, or (n, n) numpy.ndarray (default 2.)
        Line width of edges.

    edge_color : color string, or (n, n) numpy.ndarray or (n, n, 4) numpy.ndarray (default: 'k')
        Edge color. Can be a single color format string, or
        a numeric array with the first two dimensions matching the adjacency matrix.
        If a single float is specified for each edge, the values will be mapped to
        colors using the edge_cmap and edge_vmin,edge_vmax parameters.
        If a (n, n, 4) numpy.ndarray is passed in, the last dimension is
        interpreted as an RGBA tuple, that requires no further parsing.

    edge_cmap : Matplotlib colormap or None (default None)
        Colormap for mapping intensities of edges.
        Ignored if edge_color is a string or a (n, n, 4) numpy.ndarray.

    edge_vmin, edge_vmax : float, float (default None, None)
        Minimum and maximum for edge colormap scaling.
        Ignored if edge_color is a string or a (n, n, 4) numpy.ndarray.

    edge_alpha : float (default 1.)
        The edge transparency,
        Ignored if edge_color is a (n, n, 4) numpy.ndarray.

    ax : matplotlib.axis instance or None (default None)
        Draw the graph in the specified Matplotlib axis.

    arc_above: bool, optional (default True)
        If True, draw edges arcing above x-axis.

    draw_arrows : bool, optional (default True)
        If True, draw edges with arrow heads.

    Returns
    -------
    artists: dict
        Dictionary mapping edges to matplotlib.patches.FancyArrow artists.
        The dictionary keys are of the format: (source index, target index).

    """

    if not ax:
        ax = plt.gca()

    number_of_nodes = len(node_positions)

    if isinstance(edge_width, (int, float)):
        edge_width = edge_width * np.ones_like(adjacency_matrix, dtype=np.float)

    if isinstance(edge_color, np.ndarray):
        if (edge_color.ndim == 3) and (edge_color.shape[-1] == 4): # i.e. full RGBA specification
            pass
        else: # array of floats that need to parsed
            edge_color = _parse_color_input(adjacency_matrix.size,
                                            edge_color.ravel(),
                                            cmap=edge_cmap,
                                            vmin=edge_vmin,
                                            vmax=edge_vmax,
                                            alpha=edge_alpha)
            edge_color = edge_color.reshape([number_of_nodes, number_of_nodes, 4])
    else: # single float or string
        edge_color = _parse_color_input(adjacency_matrix.size,
                                        edge_color,
                                        cmap=edge_cmap,
                                        vmin=edge_vmin,
                                        vmax=edge_vmax,
                                        alpha=edge_alpha)
        edge_color = edge_color.reshape([number_of_nodes, number_of_nodes, 4])

    sources, targets = np.where(~np.isnan(adjacency_matrix))
    edge_list = list(zip(sources.tolist(), targets.tolist()))

    # order if necessary
    if edge_zorder is None:
        pass
    else:
        order = np.argsort(edge_zorder[sources, targets])
        edge_list = [edge_list[ii] for ii in order]

    # plot edges
    artists = dict()
    for (source, target) in edge_list:
        artists[(source, target)] = _add_edge(source,
                                              target,
                                              node_positions,
                                              node_artists=node_artists,
                                              edge_width=edge_width[source, target],
                                              edge_color=edge_color[source, target],
                                              arc_above=arc_above,
                                              draw_arrows=draw_arrows,
                                              ax=ax)

    return artists

def _adjacency_to_list(adjacency_matrix):
    sources, targets = np.where(~np.isnan(adjacency_matrix))
    edge_list = zip(sources.tolist(), targets.tolist())
    return edge_list

def _add_edge(source, target,
              node_positions,
              node_artists,
              edge_width,
              edge_color,
              arc_above,
              draw_arrows,
              ax):

    source_pos = node_positions[source]
    target_pos = node_positions[target]

    # base radius expressed as a fraction of the half-distance between nodes
    rad = 1.

    # make sure that edges going right to left are plotted on correct side of x-axis;
    # prevent bidirectional connections to be plotted on top of each other
    # by scaling them slightly differently
    if target_pos[0] - source_pos[0] > 0:
        rad *= 1.1
    else:
        rad *= -0.9

    # negative radius for clockwise curve
    if arc_above:
        rad *= -1

    if draw_arrows:
        arrowstyle = "fancy,head_length={},head_width={},tail_width={}".format(2*edge_width, 3*edge_width, edge_width)
    else: # make arrow heads really small
        arrowstyle = "fancy,head_length={},head_width={},tail_width={}".format(1e-10, 1e-10, edge_width)

    # stop edges from being plotted on top or bottom of node artists;
    if node_artists:
        patchA = node_artists[(source, 'edge')]
        patchB = node_artists[(target, 'edge')]

    arrow = FancyArrowPatch(source_pos, target_pos,
                            connectionstyle="arc3,rad={}".format(rad),
                            arrowstyle=arrowstyle,
                            patchA=patchA,
                            patchB=patchB,
                            facecolor=edge_color,
                            edgecolor='none')
    ax.add_patch(arrow)

    return arrow

def _update_view(adjacency_matrix, node_positions, ax):
    """
    Patches are not registered properly
    when matplotlib sets axis limits automatically.
    This function computes and sets sensible x and y limits based on the data.
    """

    maxx, maxy = np.max(node_positions, axis=0)
    minx, miny = np.min(node_positions, axis=0)

    # maxy also depends on longest arc
    edge_list = _adjacency_to_list(adjacency_matrix)
    distances = [_get_distance(node_positions[source], node_positions[target]) for source, target in edge_list]
    max_arc_height = 1.1*np.max(distances) / 2.
    maxy += max_arc_height
    miny -= max_arc_height

    w = maxx-minx
    h = maxy-miny
    padx, pady = 0.05*w, 0.05*h

    # corners = (minx-padx, miny-pady), (maxx+padx, maxy+pady)
    # ax.update_datalim(corners)
    # ax.autoscale_view()
    ax.set_xlim(minx-padx, maxx+padx)
    ax.set_ylim(miny-pady, maxy+pady)

    ax.get_figure().canvas.draw()
    return

def _get_distance(source_pos, target_pos):
    dx = source_pos[0] - target_pos[0]
    dy = source_pos[1] - target_pos[1]
    d = np.sqrt(dx**2 + dy**2)
    return d

# verbatim in netgraph
def _parse_color_input(number_of_elements, color_spec,
                       cmap=None, vmin=None, vmax=None, alpha=1.):
    """
    Handle the mess that is matplotlib color specifications.
    Return an RGBA array with specified number of elements.

    Arguments
    ---------
    number_of_elements: int
        Number (n) of elements to get a color for.

    color_spec : color string, list of strings, a float, or a numpy.ndarray of floats
        Any valid matplotlib color specification.
        If numeric values are specified, they will be mapped to colors using the
        cmap and vmin/vmax arguments.

    cmap : matplotlib colormap (default None)
        Color map to use if color_spec is not a string.

    vmin, vmax : float, float (default None, None)
        Minimum and maximum values for normalizing colors if a color mapping is used.

    alpha : float or n-long iterable of floats (default 1.)
        Alpha values to go with the colors.

    Returns
    -------
    rgba_array : (n, 4) numpy ndarray
        Array of RGBA color specifications.

    """
    # map color_spec to either a list of strings or
    # an iterable of floats of the correct length,
    # unless, of course, they already are either of these
    if isinstance(color_spec, (float, int)):
        color_spec = color_spec * np.ones((number_of_elements), dtype=np.float)
    if isinstance(color_spec, str):
        color_spec = number_of_elements * [color_spec]

    # map numeric types using cmap, vmin, vmax
    if isinstance(color_spec[0], (float, int)):
        mapper = cm.ScalarMappable(cmap=cmap)
        mapper.set_clim(vmin, vmax)
        rgba_array = mapper.to_rgba(color_spec)
    # convert string specification to colors
    else:
        rgba_array = np.array([colorConverter.to_rgba(c) for c in color_spec])

    # Set the final column of the rgba_array to have the relevant alpha values.
    rgba_array[:,-1] = alpha

    return rgba_array

# verbatim in netgraph
def _make_pretty(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    ax.get_figure().set_facecolor('w')
    ax.set_frame_on(False)
    ax.get_figure().canvas.draw()
    return

# verbatim in netgraph
def _get_random_weight_matrix(n, p,
                              weighted=True,
                              strictly_positive=False,
                              directed=True,
                              fully_bidirectional=False,
                              dales_law=False):

    if weighted:
        w = np.random.randn(n, n)
    else:
        w = np.ones((n, n))

    if strictly_positive:
        w = np.abs(w)

    if not directed:
        w = np.triu(w)
        w[np.tril_indices(n)] = np.nan

    if directed and fully_bidirectional:
        c = np.random.rand(n, n) <= p/2
        c = np.logical_or(c, c.T)
    else:
        c = np.random.rand(n, n) <= p
    w[~c] = np.nan

    if dales_law and weighted and not strictly_positive:
        w = np.abs(w) * np.sign(np.random.randn(n))[:,None]
    return w

def test(n=20, p=0.1, ax=None, directed=True, **kwargs):
    # create two networks that are similar to each other
    # by combining a common core network with two different networks
    w1 = _get_random_weight_matrix(n, p/3., directed=directed)
    w2 = _get_random_weight_matrix(n, 2*p/3., directed=directed)
    w3 = _get_random_weight_matrix(n, p/3., directed=directed)

    for w in [w1,w2,w3]:
        w[np.isnan(w)] = 0.

    w12 = w1 + w2
    w23 = w2 + w3

    w123 = w1+w2+w3 # for plotting

    for w in [w12, w23, w123]:
        w[w==0] = np.nan

    # node_order = range(n)
    node_order = _optimize_node_order(w123)

    fig, ax = plt.subplots(1,1)

    max_val = 3.
    draw(w12, node_order=node_order, ax=ax, arc_above=True, edge_vmin=-max_val, edge_vmax=max_val)
    draw(w23, node_order=node_order, ax=ax, arc_above=False, edge_vmin=-max_val, edge_vmax=max_val)

    ax.text(0,1, 'Before', transform=ax.transAxes, fontsize=18)
    ax.text(0,0, 'After', transform=ax.transAxes, fontsize=18)
    plt.show()

    return ax

def _get_modular_weight_matrix(module_sizes, p_in=0.5, p_ex=0.1, show_plot=False):
    n = np.sum(module_sizes)
    c = np.random.rand(n,n) < p_ex
    w = np.random.randn(n,n)
    w[~c] = 0.

    start = np.cumsum(np.concatenate([[0], module_sizes]))[:-1]
    stop = np.cumsum(np.concatenate([[0], module_sizes]))[1:]

    for ii, s in enumerate(module_sizes):
        ww = np.random.randn(s,s)
        cc = np.random.rand(s,s) < p_in - p_ex
        ww[~cc] = 0.
        w[start[ii]:stop[ii], start[ii]:stop[ii]] += ww

    if show_plot:
        fig, ax = plt.subplots(1,1)
        ax.imshow(w, cmap='gray', interpolation='none')

    w[w==0] = np.nan
    return w

def test_modular_graph():
    w = _get_modular_weight_matrix([10,10], p_in=0.9, p_ex=0.1)
    optimal_order = _optimize_node_order(w)
    random_order = np.random.permutation(optimal_order)

    fig, ax = plt.subplots(1,1)

    max_val = 3.
    draw(w, node_order=optimal_order, ax=ax, arc_above=True, edge_vmin=-max_val, edge_vmax=max_val)
    draw(w, node_order=random_order, ax=ax, arc_above=False, edge_vmin=-max_val, edge_vmax=max_val)
    plt.show()

    return
