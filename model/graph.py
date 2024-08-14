import torch
import torch.nn as nn
from utils.util import normalized_laplacian

class GraphConvLayer(nn.Module):
    """ GCN Layer implementation of https://arxiv.org/abs/1609.02907
    """

    def __init__(self, dim_in: int, dim_out: int, laplacian) -> None:
        super(GraphConvLayer, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out

        self._build_net(laplacian)
        self._init_parameters()


    def _build_net(self, laplacian: torch.Tensor) -> None:
        # self.laplacian shape: (N, N)
        self.register_buffer('laplacian', laplacian)
        self.weights = nn.Parameter(
            torch.Tensor(self.dim_in, self.dim_out)
        )
        self.bias = nn.Parameter(
            torch.Tensor(self.dim_out)
        )

    def _init_parameters(self):
        nn.init.xavier_uniform_(self.weights, gain=nn.init.calculate_gain("tanh"))


    def forward(self, inputs: torch.FloatTensor):
        # B: batch size
        # N: number of nodes
        # T: sequence length
        # C: channels
        B, N, T, C = inputs.shape

        # (B,N,T,C) -> (N,B,T,C)
        inputs = inputs.transpose(0, 1)

        # (N,B,T,C) -> (N, B*T*C)
        inputs = inputs.reshape((N, B*T*C))

        # AX
        # shape: (N, B*T*C)
        ax = self.laplacian @ inputs
        # (N, B*T*C) -> (N*B*T, C)
        ax = ax.reshape((N*B*T, C))

        # AXW
        # H: hidden dimension
        # shape: (N*B*T, H)
        axw_b = ax @ self.weights + self.bias

        # activation
        outputs = torch.tanh(axw_b)

        # reshape back to original shape
        outputs = outputs.reshape((N,B,T,self.dim_out))

        # (N,B,T,H) -> (B,N,T,H)
        outputs = outputs.transpose(0, 1)
        return outputs


class GraphConvGRULayer(nn.Module):
    """ Alternative implementation of the GraphConv Layer for the decoupled GRU
        This implementation takes hidden states + inputs and does not apply
        activation
    """

    def __init__(self, laplacian, dim_in: int, dim_out: int, bias: float = 0.0):
        super(GraphConvGRULayer, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self._bias_init_value = bias

        self.register_buffer("laplacian", laplacian)
        self.weights = nn.Parameter(torch.FloatTensor(self.dim_in, self.dim_out))
        self.biases = nn.Parameter(torch.FloatTensor(self.dim_out))
        self.reset_parameters()


    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights)
        nn.init.constant_(self.biases, self._bias_init_value)


    def forward(self, inputs: torch.FloatTensor):

        # B: batch size
        # N: number of nodes
        # C: channels
        B, N, C = inputs.shape

        # inputs = [x, h] (B,N,H) -> (N,B,H)
        inputs = inputs.transpose(0, 1)
        # X = [x, h] (N,B,H) -> (N,B*H)
        inputs = inputs.reshape((N, B*C))

        # A[x, h] (N,B*H)
        AX = self.laplacian @ inputs
        # A[x, h] (N,B*H) -> (N*B,H)
        AX = AX.reshape((N*B, C))

        # O: dim_out
        # A[x, h]W + b (N*B, O)
        AXW_B = AX @ self.weights + self.biases

        # A[x, h]W + b (N*B,O) -> (N,B,O)
        outputs = AXW_B.reshape((N, B, self.dim_out))

        # (N,B,O) -> (B,N,O)
        outputs = outputs.transpose(0, 1)
        return outputs