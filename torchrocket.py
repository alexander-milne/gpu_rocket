import torch
import torch.nn as  nn
import torch.nn.functional as F
import numpy as np
from numba import njit

class TorchRocket(torch.nn.Module):
    def __init__(self,kernels=None,num_timesteps=None,num_channels=None,num_kernels=10_000,ppv_ver='softPPV',softppv_param=7.):
        super(TorchRocket, self).__init__()
        if kernels is None and (num_timesteps is None or num_channels is None):
            raise ValueError('Must provide either kernels or num_timesteps and num_channels')
        self.convs = nn.ModuleList()
        self.torch_channel_indices = nn.ParameterList()
        if kernels is None:
            self._build_layers_from_scratch(num_timesteps,num_channels,num_kernels=num_kernels)
        else:
            self._build_layers_with_kernels(kernels)
        self.maxpool = nn.AdaptiveMaxPool1d((1))
        if ppv_ver=='softPPV':
            self._ppv=self.softppv
            self.softppv_param = softppv_param
        elif ppv_ver=='heaviside_PPV':
            self._ppv=self.heaviside_PPV

    def _build_layers_from_scratch(self,num_timesteps,num_channels=1,kernel_lengths_possibilities=[7,9,11],num_kernels=10):
        biases=torch.zeros(num_kernels, 1)
        torch.nn.init.uniform_(biases, a=0.0, b=1.0)
        kernel_lengths = [kernel_lengths_possibilities[x] for x in torch.randint(0, 3, (num_kernels,))]
        possible_channels = torch.IntTensor([x for x in range(num_channels)])
        kernel_channels = [possible_channels[torch.randperm(num_channels)[:n_select]] for n_select in torch.randint(1, num_channels, (num_kernels,))]
        self.torch_channel_indices = nn.ParameterList([nn.Parameter(x,requires_grad=False) for x in kernel_channels])
        weights = [torch.randn(1, kernel_channels[i].shape[0], kernel_lengths[i]) for i in range(len(kernel_lengths))]
        for c_weight in weights:
                for channel in range(c_weight.shape[1]):
                    c_weight[0,channel,:]=c_weight[0,channel,:]-c_weight[0,channel,:].mean()
        
        dilations = [2 ** torch.FloatTensor(1).uniform_(0,torch.log2(torch.FloatTensor([(num_timesteps - 1) / (k_len - 1)])[0])) for k_len in kernel_lengths]
        dilations = [torch.IntTensor([int(dilation)]) for dilation in dilations]
        for i in range(len(kernel_lengths)):
            # random choice of padding 'same' or 'valid'
            padding = ['same','valid'][torch.randint(0, 2, (1,))]
            self.convs.append(
                torch.nn.Conv1d(in_channels=kernel_channels[i].shape[0], out_channels=1, kernel_size=kernel_lengths[i], padding=padding, dilation=dilations[i])
            )
            self.convs[i].weight.data = torch.nn.Parameter(weights[i])
            self.convs[i].bias.data = torch.nn.Parameter(biases[i])
            
    def _build_layers_with_kernels(self,kernels):
        (weights,lengths,biases,dilations,paddings,
        num_channel_indices,channel_indices) = kernels

        start=0
        start_weights=0

        for i in range(len(dilations)):
            stop = start + num_channel_indices[i]
            stop_weights = start_weights + (num_channel_indices[i] * lengths[i]) #
            padding='valid' if paddings[i]==0 else 'same'
            self.convs.append(
                torch.nn.Conv1d(num_channel_indices[i], 1, kernel_size=lengths[i], padding=padding, dilation=dilations[i])
            )
            self.torch_channel_indices.append(nn.Parameter(torch.IntTensor(channel_indices[start:stop].copy()),requires_grad=False))
            reshaped_weights=np.array([weights[start_weights:stop_weights]]
                ).reshape(self.convs[-1].weight.data.shape)
            weights_this_kernel = torch.Tensor(reshaped_weights)
            self.convs[-1].weight.data = nn.Parameter(weights_this_kernel)
            self.convs[-1].bias = nn.Parameter(torch.Tensor([biases[i]]))
            start = stop
            start_weights = stop_weights    

    def softppv(self,x):
        return torch.mean(torch.sigmoid((self.softppv_param*x)-3.),-1)

    def heaviside_PPV(self,x):
        return torch.mean(torch.heaviside(x,torch.tensor(0.)),-1)

    def forward(self, x):
        outs=[]
        for i,kernel in enumerate(self.convs):
            _x = torch.index_select(x, dim=-2, index=self.torch_channel_indices[i])
            _x = kernel(_x)
            outs.append(self._ppv(_x))
            outs.append(torch.max(_x, dim=-1)[0])
        out = torch.cat(outs, axis=-1)
        return out


@njit(
    "Tuple((float32[:],int32[:],float32[:],int32[:],int32[:],int32[:],"
    "int32[:]))(int32,int32,int32,optional(int32))",
    cache=True,
)
def _generate_kernels_sktime_32bit_version(n_timepoints, num_kernels, n_columns, seed):
    if seed is not None:
        np.random.seed(seed)

    candidate_lengths = np.array((7, 9, 11), dtype=np.int32)
    lengths = np.random.choice(candidate_lengths, num_kernels)

    num_channel_indices = np.zeros(num_kernels, dtype=np.int32)
    for i in range(num_kernels):
        limit = min(n_columns, lengths[i])
        num_channel_indices[i] = 2 ** np.random.uniform(0, np.log2(limit + 1))

    channel_indices = np.zeros(num_channel_indices.sum(), dtype=np.int32)

    weights = np.zeros(
        np.int32(
            np.dot(lengths.astype(np.float32), num_channel_indices.astype(np.float32))
        ),
        dtype=np.float32,
    )
    biases = np.zeros(num_kernels, dtype=np.float32)
    dilations = np.zeros(num_kernels, dtype=np.int32)
    paddings = np.zeros(num_kernels, dtype=np.int32)

    a1 = 0  # for weights
    a2 = 0  # for channel_indices

    for i in range(num_kernels):

        _length = lengths[i]
        _num_channel_indices = num_channel_indices[i]

        _weights = np.random.normal(0, 1, _num_channel_indices * _length)

        b1 = a1 + (_num_channel_indices * _length)
        b2 = a2 + _num_channel_indices

        a3 = 0  # for weights (per channel)
        for _ in range(_num_channel_indices):
            b3 = a3 + _length
            _weights[a3:b3] = _weights[a3:b3] - _weights[a3:b3].mean()
            a3 = b3

        weights[a1:b1] = _weights

        channel_indices[a2:b2] = np.random.choice(
            np.arange(0, n_columns), _num_channel_indices, replace=False
        )

        biases[i] = np.random.uniform(-1, 1)

        dilation = 2 ** np.random.uniform(
            0, np.log2((n_timepoints - 1) / (_length - 1))
        )
        dilation = np.int32(dilation)
        dilations[i] = dilation

        padding = ((_length - 1) * dilation) // 2 if np.random.randint(2) == 1 else 0
        paddings[i] = padding

        a1 = b1
        a2 = b2

    return (
        weights,
        lengths,
        biases,
        dilations,
        paddings,
        num_channel_indices,
        channel_indices,
    )

