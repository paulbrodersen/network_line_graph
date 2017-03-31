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

from matplotlib.patches import FancyArrowPatch, Circle

def draw_comparison():
    pass

def draw():
    pass

def draw_nodes():
    pass

def draw_edges(edges, node_order=None, ax=None, arc_above=True, **arrow_kwargs):
    """
    Arrange nodes on a line and plot the edges between them as Bezier curves.

    Arguments:
    ----------
    edges      -- list of (source, target) tuples
    node_order -- list of nodes (default None)
    ax         -- (default None)

    Returns:
    --------
    ax         -- matplotlib.axes._subplots.AxesSubplot instance

    """

    if not ax:
        fig, ax = plt.subplots(1,1)

    if not node_order:
        node_order = np.unique(np.array(edges))

    # compute node positions
    n = len(node_order)
    node2pos = dict(zip(node_order, zip(np.arange(n), np.zeros((n)))))

    # plot edges
    for (source, target) in edges:
        _add_edge(node2pos[source], node2pos[target], ax, arc_above, **arrow_kwargs)

    # set sensible x and y limits
    _update_view(edges, node2pos, ax)

    return ax

def _add_edge(source_pos, target_pos, ax, arc_above, **arrow_kwargs):
    # base radius expressed as a fraction of distance between nodes
    rad = 1.

    # make sure that edges going right to left are plotted on correct side of x-axis;
    # prevent bidirectional connections to be plotted on top of each other by scaling them slightly differently
    if target_pos[0] - source_pos[0] > 0:
        rad *= 1.1
    else:
        rad *= -0.9

    # negative radius for clockwise curve
    if arc_above:
        rad *= -1

    # draw
    arrow = FancyArrowPatch(source_pos, target_pos,
                            connectionstyle="arc3,rad={}".format(rad),
                            **arrow_kwargs)
    ax.add_patch(arrow)

    return ax

def _update_view(edges, node2pos, ax):
    """
    Patches are not registered properly
    when matplotlib sets axis limits automatically.
    This function computes and sets sensible x and y limits based on the data.
    """

    node_positions = np.array(node2pos.values())
    maxx, maxy = np.max(node_positions, axis=0)
    minx, miny = np.min(node_positions, axis=0)

    # maxy also depends on longest arc
    max_arc_height = np.max([_get_distance(node2pos[source], node2pos[target]) for source, target in edges]) / 2.
    maxy += max_arc_height
    # miny -= max_arc_height

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

def test():

    # create a sparse graph adjacency matrix
    n = 10
    p = 0.25
    a = np.random.rand(n,n) < p

    # get edge list
    edges = zip(*np.where(a))

    w = 2
    arrow_kwargs = dict(arrowstyle="fancy,head_length={},head_width={},tail_width={}".format(2*w, 3*w, w),
                        linewidth=w,
                        facecolor='red',
                        edgecolor='none')

    # plot
    fig, ax = plt.subplots(1,1)
    draw_edges(edges, node_order=range(n), ax=ax, **arrow_kwargs)
    plt.show()

    return
