import torch
import torch.nn as nn
from model.graph import GraphConvGRULayer

class GCNGRUCell(nn.Module):
    """ This part implements the GRU weight calculation, applying GCN on each recurrance
    """

    def __init__(self, laplacian, dim_in: int, dim_hidden: int):
        super(GCNGRUCell, self).__init__()
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden

        self._build_net(laplacian)


    def _build_net(self, laplacian):
        # this layer combines the input and hidden, and output a concat that'll be broken into two
        self.graph_conv1 = GraphConvGRULayer(laplacian, self.dim_in + self.dim_hidden, self.dim_hidden * 2, bias=1.0)

        self.graph_conv2 = GraphConvGRULayer(laplacian, self.dim_in + self.dim_hidden, self.dim_hidden)


    def forward(self, inputs, hidden_state):
        # shapes of `inputs`: (B, N, dim_in)
        # shape of `hidden_state`: (B, N, dim_hidden)

        # concatenation: (B, N, dim_in + dim_hidden)
        concatenation = torch.cat((inputs, hidden_state), dim=2)

        # [r, u] = sigmoid(A[x, h]W + b)
        # [r, u] (batch_size, num_nodes * (2 * dim_in))
        concatenation = torch.sigmoid(self.graph_conv1(concatenation))

        # r (batch_size, num_nodes, dim_in)
        # u (batch_size, num_nodes, dim_in)
        r, u = torch.chunk(concatenation, chunks=2, dim=2)

        # c = tanh(A[x, (r * h)W + b])
        # c (batch_size, num_nodes * dim_in)
        rh = torch.cat((inputs, r * hidden_state), dim=2)
        c = torch.tanh(self.graph_conv2(rh)) # hadamard product

        # h := u * h + (1 - u) * c
        # h (batch_size, num_nodes * num_gru_units)
        new_hidden_state = u * hidden_state + (1.0 - u) * c
        return new_hidden_state, new_hidden_state



class GCNGRU(nn.Module):
    def __init__(self, dim_in, laplacian):
        super(GCNGRU, self).__init__()

        # at this point it is important that the two matches
        self.dim_in = dim_in
        self.dim_hidden = dim_in

        self.tgcn_cell = GCNGRUCell(laplacian, self.dim_in, self.dim_hidden)


    def forward(self, inputs: torch.Tensor):
        # shape of input
        # B: batch size
        # N: number of slope units
        # T: sequence length
        # C: number of input channels (which is dim_rain + dim_geo)
        B, N, T, C = inputs.shape

        # assert self._input_dim == num_nodes

        hidden_state = torch.zeros((B, N, self.dim_hidden)).type_as(inputs)
        output = None

        # unravel and apply sequential data ourselves
        for i in range(T):
            output, hidden_state = self.tgcn_cell(inputs[:, :, i, :], hidden_state)
            output = output.reshape((B, N, self.dim_hidden))
        return output