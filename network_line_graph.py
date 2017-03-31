#!/usr/bin/env python
# -*- coding: utf-8 -*-

# network_line_graph: visualisation to compare different states of a network.

# Copyright (C) 2016 Paul Brodersen <paulbrodersen+netgraph@gmail.com>

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
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.cbook as cb
from matplotlib.colors import colorConverter, Colormap
from matplotlib.patches import FancyArrowPatch, Circle

def draw_comparison():
    pass

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

    node_positions : (n, 2) numpy.ndarray
        (x, y) node coordinates.

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
        kwargs.setdefault('edge_vmin', 0.)
        kwargs.setdefault('edge_vmax', 1.)
        kwargs.setdefault('edge_cmap', 'RdGy')
        kwargs.setdefault('edge_zorder', edge_zorder)

    number_of_nodes = adjacency_matrix.shape[0]
    if node_order is None:
        node_order = np.arange(number_of_nodes)

    node_positions = _get_positions(node_order)

    draw_edges(adjacency_matrix, node_positions, **kwargs)
    # draw_nodes(node_positions, **kwargs)

    if node_labels is not None:
        draw_node_labels(node_positions, node_labels)

    # Patches are not registered properly
    # when matplotlib sets axis limits automatically.
    # So we need to do this manually.
    _update_view(adjacency_matrix, node_positions, ax)

    # remove superfluous ink
    _make_pretty(ax)

    return

def draw_nodes():
    pass

def _get_positions(node_order):
    n = len(node_order)
    node_positions = np.array(zip(node_order, np.zeros((n))))
    return node_positions

def draw_edges(adjacency_matrix,
               node_positions,
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

    node_size : scalar or (n,) numpy.ndarray (default 0.)
        Size (radius) of nodes. Used to offset edges when drawing arrow heads,
        such that the arrow heads are not occluded.
        Nota bene: in draw_nodes() the node_size default is 3.!
        If draw_nodes() and draw_edges() are called independently,
        make sure to set this variable to the same value.

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
    edge_list = zip(sources.tolist(), targets.tolist())

    # order if necessary
    if edge_zorder is None:
        pass
    else:
        order = np.argsort(edge_zorder[sources, targets])
        edge_list = [edge_list[ii] for ii in order]

    # plot edges
    artists = dict()
    for (source, target) in edge_list:
        artists[(source, target)] = _add_edge(node_positions[source],
                                              node_positions[target],
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

def _add_edge(source_pos, target_pos, edge_width, edge_color, arc_above, draw_arrows, ax):
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

    # draw
    if draw_arrows:
        arrowstyle = "fancy,head_length={},head_width={},tail_width={}".format(2*edge_width, 3*edge_width, edge_width)
    else:
        arrowstyle = "fancy,head_length={},head_width={},tail_width={}".format(1e-10, 1e-10, edge_width)

    arrow = FancyArrowPatch(source_pos, target_pos,
                            connectionstyle="arc3,rad={}".format(rad),
                            arrowstyle=arrowstyle,
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
    max_arc_height = np.max(distances) / 2.
    maxy += max_arc_height
    miny -= max_arc_height

    w = maxx-minx
    h = maxy-miny
    padx, pady = 0.05*w, 0.05*h
    corners = (minx-padx, miny-pady), (maxx+padx, maxy+pady)

    ax.update_datalim(corners)
    ax.autoscale_view()
    ax.get_figure().canvas.draw()
    return

def _get_distance(source_pos, target_pos):
    dx = source_pos[0] - target_pos[0]
    dy = source_pos[1] - target_pos[1]
    d = np.sqrt(dx**2 + dy**2)
    return d

# verbatim copy of netgraph._parse_color_input
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

def _make_pretty(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    # ax.set_aspect('equal') # <- only difference to version in netgraph
    ax.get_figure().set_facecolor('w')
    ax.set_frame_on(False)
    ax.get_figure().canvas.draw()
    return

def test(n=20, p=0.15, ax=None, directed=True, **kwargs):
    w = _get_random_weight_matrix(n, p, directed=directed)

    # plot
    fig, ax = plt.subplots(1,1)
    draw(w, node_order=range(n), ax=ax)
    plt.show()

    return ax

# verbatim from netgraph
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
