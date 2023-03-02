import torch
from torch_geometric.nn import GCNConv


class LabeledConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        in_channels - dimension of node features in input
        out_channels - dimension of node features in output
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 4 different edge convolutions
        self.conv1 = GCNConv(in_channels, out_channels, bias=True)
        self.conv2 = GCNConv(in_channels, out_channels, bias=True)
        self.conv3 = GCNConv(in_channels, out_channels, bias=True)
        self.conv4 = GCNConv(in_channels, out_channels, bias=True)
        
        self.params = torch.nn.ParameterList([torch.nn.Parameter(torch.randn(self.out_channels)) for _ in range(6)])
    
    def forward(self, x, edges00, edges01, edges10, edges11, type0_idx, type1_idx):
        """
        torch tensors of shape 2 x smthng that lists all edges:
        edges00 - between nodes of type 0
        edges01 - from type 0 to type 1
        edges10 - from type 1 to type 0
        edges11 - between nodes of type 1

        type0_idx, type1_idx - torch tensors of shape n x 1 
        with '1's at positions of type 0 / type 1 nodes
        """
        
        x0 = torch.mul(type0_idx, x.t()[0]).repeat(self.out_channels, 1).t()
        x1 = torch.mul(type1_idx, x.t()[0]).repeat(self.out_channels, 1).t()
        
        # compute messages from type 1 nodes and type 2 nodes separately
        x00 = torch.zeros(x.shape[0], self.out_channels)
        x01 = torch.zeros(x.shape[0], self.out_channels)
        x10 = torch.zeros(x.shape[0], self.out_channels)
        x11 = torch.zeros(x.shape[0], self.out_channels)

        if edges00.shape[0] != 0:
            x00 = self.conv1(x, edges00)
        if edges01.shape[0] != 0:
            x01 = self.conv2(x, edges01)
        if edges10.shape[0] != 0:
            x10 = self.conv3(x, edges10)
        if edges11.shape[0] != 0:
            x11 = self.conv4(x, edges11)

        # linear combination of 6 equivariant layers
        out = (torch.einsum('ij,j->ij', x0, self.params[0]) + torch.einsum('ij,j->ij', x1, self.params[1]) +
            torch.einsum('ij,j->ij', x00, self.params[2]) + torch.einsum('ij,j->ij', x01, self.params[3]) +
            torch.einsum('ij,j->ij', x10, self.params[4]) + torch.einsum('ij,j->ij', x11, self.params[5]))
        
        return out