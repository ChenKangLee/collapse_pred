import torch
import torch.nn as nn
import math

class GraphConvLayer(nn.Module):
    """ GCN Layer implementation of https://arxiv.org/abs/1609.02907
        credits: https://github.com/tkipf/pygcn
    """

    def __init__(self, dim_in, dim_out, use_bias=True):
        super(GraphConvLayer, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.use_bias = use_bias

        self._build_net()
        self._init_parameters()


    def _build_net(self):
        # using nn.Parameter so we can control the use of bias
        # as nn.Linear applies bias by default
        self.weight = nn.Parameter(torch.empty((self.dim_in, self.dim_out)))

        if self.use_bias:
            self.bias = nn.Parameter(torch.empty(self.dim_out))
        else:
            self.register_parameter('bias', None)


    def _init_parameter(self):
        # heuritic initialization of weight pre-xavier
        # TODO: try xavier if not seeing good results
        stdv = 1. / math.sqrt(self.weight.size(1))

        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)


    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
        

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'