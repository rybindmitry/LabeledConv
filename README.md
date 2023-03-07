# Labeled Convolutions
Implementation of Graph Convolution Layer for graphs with nodes of two different types.


# Usage
The code provides LabeledConv class, which can be directly used instead of the usual GCNConv layer in your Graph Neural Network. Instead of

y = GCNConv(x, edge_data)

use the line

y = LabeledConv(x, node_type0, node_type1, edge_data00, edge_data01, edge_data10, edge_data11)

To make LabeledConv flexible, the implementation asks to provide vectors of 1's identifying nodes of two types, as well as edge data. 

# Experiments


