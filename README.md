# Labeled Convolutions
Implementation of Graph Convolution Layer for graphs with nodes of two different types.

![two_types_1](https://user-images.githubusercontent.com/42006378/223395849-373f022e-f345-452f-98e5-3055cab44855.png)


# Usage
The code provides LabeledConv class, which can be directly used instead of the usual GCNConv layer in your Graph Neural Network. Instead of

```python
y = GCNConv(x, edge_data)
```

use the line

```python
y = LabeledConv(x, node_type0, node_type1, edge_data00, edge_data01, edge_data10, edge_data11)
```

To make LabeledConv flexible, the implementation asks to provide vectors of 1's identifying nodes of two types, as well as edge data. 

# Experiments

## Subgraph properties
In some data we have to determine properties of subgraph $S$ of a given graph $G$. It follows that there are nodes of two types: nodes from subgraph $S$ and nodes not from $S$.

## Minimal Vertex Cover
During a decision process on a graph, some nodes are selected. This turns their 1-hop neighbours into covered nodes. In each step of the decision process there are nodes of two types: covered / not covered.

