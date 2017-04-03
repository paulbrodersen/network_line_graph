# Network Line Graph

Visualise a network by assigning nodes a position along the
x-axis and then drawing the edges between them as Bezier curves.

Such a visualisation is particularly useful to compare two different
network states, as a second network state can be drawn in the same way
below the x-axis. The symmetry around the x-axis accentuates changes
in the network structure.

![alt tag](./example.png)

## Example:

``` python
# make a weighted random graph
n = 20 # number of nodes
p = 0.1 # connection probability
a = np.random.rand(n,n) < p # adjacency matrix
w = np.random.randn(n,n) # weight matrix
w[~a] = np.nan

# initialise figure
fig, ax = plt.subplots(1,1)

import network_line_graph as nlg
nlg.draw(w, arc_above=True, ax=ax)
```
