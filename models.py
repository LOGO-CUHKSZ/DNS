import torch
import torch.nn as nn

import numpy as np

from torch import Tensor
from typing import List
from entmax import sparsemax

__all__ = [
    'CTGRU',
    'DNS',
    'SparseDNS',
    'DisDNS',
    'NeuralCDE',
    'RIM'
]

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class CTGRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, tau: int = 1, M: int = 8) -> None:
        """M should be in the range of {4, 5, ..., 9}. """
        super(CTGRU, self).__init__()
        if M < 3 or M > 9:
            raise ValueError('M should be in the range of {4, 5, ..., 9}.')
        self.units = hidden_size
        self.M = M
        self.state_size = hidden_size * M
        
        self.ln_tau_table = torch.empty(self.M, device=DEVICE)
        self.tau_table = torch.empty(self.M, device=DEVICE)
        tau = torch.tensor(tau, device=DEVICE)
        for i in range(self.M):
            self.ln_tau_table[i] = torch.log(tau)
            self.tau_table[i] = tau
            tau = tau * (10.0 ** 0.5)

        self.retrieval_layer = nn.Linear(input_size + hidden_size, M * hidden_size)
        self.detect_layer = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Tanh()
        )
        self.update_layer = nn.Linear(input_size + hidden_size, M * hidden_size)

        self.decoder = nn.Linear(hidden_size * self.M, output_size)

    def forward(self, t: Tensor, x: Tensor) -> Tensor:
        # t.shape == (batch_size, n_take)
        # x.shape == (batch_size, n_take, input_size)

        h_hat = torch.zeros((t.size(0), self.units, self.M), device=t.device)
        
        for i in range(t.size(1)):
            h = torch.sum(h_hat, dim=2)
            fused_input = torch.cat([x[:, i, :], h], dim=-1)
            ln_tau_r = self.retrieval_layer(fused_input).reshape(-1, self.units, self.M)
            
            sf_input_r = - torch.square(ln_tau_r - self.ln_tau_table)
            rki = nn.Softmax(dim=2)(sf_input_r)

            q_input = torch.sum(rki * h_hat, dim=2)
            reset_value = torch.cat([x[:, i, :], q_input], dim=1)
            qk = self.detect_layer(reset_value).unsqueeze(dim=-1)

            ln_tau_s = self.update_layer(fused_input).reshape(-1, self.units, self.M)
            sf_input_s = - torch.square(ln_tau_s - self.ln_tau_table)
            ski = nn.Softmax(dim=2)(sf_input_s)

            base_term = (1 - ski) * h_hat + ski * qk
            exp_term = torch.exp(- t[:, i].repeat(self.M).reshape(-1, self.M) / self.tau_table).reshape(-1, 1, self.M)

            h_hat = base_term * exp_term

        return self.decoder(h_hat.reshape(-1, self.units * self.M))

class NaturalCubicSpline:
    def __init__(self, times: Tensor, coeffs: List[Tensor], **kwargs) -> None:
        super(NaturalCubicSpline, self).__init__(**kwargs)
        (a, b, two_c, three_d) = coeffs
        self._times = times # times.shape == (batch_size, n_take)
        self._a = a
        self._b = b
        # as we're typically computing derivatives, we store the multiples of these coefficients that are more useful
        self._two_c = two_c
        self._three_d = three_d
        self.range = torch.arange(0, self._times.size(0))

    def _interpret_t(self, t: Tensor) -> Tensor:
        maxlen = self._b.size(-2) - 1
        index = (t > self._times).sum(dim=1) - 1 # index.size == (batch_size)
        index = index.clamp(0, maxlen)  # clamp because t may go outside of [t[0], t[-1]]; this is fine
        # will never access the last element of self._times; this is correct behaviour
        fractional_part = t - self._times[self.range, index]
        return fractional_part.unsqueeze(dim=1), index

    def evaluate(self, t: Tensor) -> Tensor:
        """Evaluates the natural cubic spline interpolation at a point t, which should be a scalar tensor."""
        fractional_part, index = self._interpret_t(t)
        inner = 0.5 * self._two_c[self.range, index, :] + self._three_d[self.range, index, :] * fractional_part / 3
        inner = self._b[self.range, index, :] + inner * fractional_part
        return self._a[self.range, index, :] + inner * fractional_part

    def derivative(self, t: Tensor) -> Tensor:
        """Evaluates the derivative of the natural cubic spline at a point t, which should be a scalar tensor."""
        fractional_part, index = self._interpret_t(t)
        inner = self._two_c[self.range, index, :] + self._three_d[self.range, index, :] * fractional_part
        deriv = self._b[self.range, index, :] + inner * fractional_part
        return deriv

class CDEFunc(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int, num_layers: int = 2) -> None:
        super(CDEFunc, self).__init__()
        if num_layers < 2:
            raise ValueError(f'num_layers={num_layers} should be no less than 2.')
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.nn = [nn.Linear(hidden_channels, hidden_channels), nn.ReLU()]
        for _ in range(num_layers - 2):
            self.nn.append(nn.Linear(hidden_channels, hidden_channels))
            self.nn.append(nn.ReLU())
        self.nn.append(nn.Linear(hidden_channels, input_channels * hidden_channels))
        self.nn.append(nn.Tanh())

        self.nn = nn.Sequential(*self.nn).to(DEVICE)

    def forward(self, z: Tensor) -> Tensor:
        z = self.nn(z)
        return z.view(*z.shape[:-1], self.hidden_channels, self.input_channels)

class GroupLinear_1_to_n(nn.Module):
    def __init__(self, input_size: int, output_size: int, n_block: int) -> None:
        super(GroupLinear_1_to_n, self).__init__()
        self.linear = nn.Linear(input_size, 2 * input_size)
        self.w = nn.Parameter(0.01 * torch.rand(n_block, 2 * input_size, output_size))
        self.b = nn.Parameter(torch.zeros(n_block, output_size))

    def forward(self, x: Tensor) -> Tensor:
        # x.shape == (batch_size, input_size)
        # return.shape == (batch_size, n_blocks, hidden_size)
        return torch.matmul(self.linear(x).relu(), self.w).permute(1, 0, 2) + self.b

class GroupLinear_n_to_m(nn.Module):
    def __init__(self, input_size: int, output_size: int, n_block: int) -> None:
        super(GroupLinear_n_to_m, self).__init__()
        self.n_block = n_block
        self.linear = nn.Linear(input_size, output_size)
        self.w = nn.Parameter(0.01 * torch.rand(n_block, output_size, output_size))
        self.b = nn.Parameter(torch.zeros(n_block, output_size))

    def forward(self, x: Tensor) -> Tensor:
        # x.shape == (batch_size, n_take, input_size)
        x = self.linear(x).relu() # x.shape == (batch_size, n_take, output_size)
        # return.shape == (batch_size, n_blocks, n_take, output_size)
        
        return torch.matmul(x.expand(self.n_block, -1, -1, -1).permute(1, 0, 2, 3), self.w) + self.b.expand(x.size(1), -1, -1).permute(1, 0, 2)

class DNS(nn.Module):
    def __init__(self, input_size: int = 28, hidden_size: int = 16, output_size: int = 10, n_blocks: int = 6, num_layers: int = 2) -> None:
        super(DNS, self).__init__()
        self.hidden_size = hidden_size
        self.n_blocks = n_blocks
        
        self.x_encoder = GroupLinear_n_to_m(input_size, 2 * input_size, n_blocks)

        self.h_encoder = GroupLinear_1_to_n(input_size, hidden_size, n_blocks)
        self.h_decoder = nn.Linear(n_blocks * hidden_size, output_size)

        self.hidden_func = CDEFunc(2 * input_size, hidden_size, num_layers)

        self.key_w_T = nn.Parameter(0.01 * torch.randn(n_blocks * n_blocks, hidden_size))
        self.query_w = nn.Parameter(0.01 * torch.randn(hidden_size, n_blocks * n_blocks))

        self.key_b = nn.Parameter(torch.zeros(n_blocks))
        self.query_b = nn.Parameter(torch.zeros(n_blocks))

    def transpose_for_scores(self, x: Tensor) -> Tensor:
        new_x_shape = x.size()[:-1] + (1, self.n_blocks * self.n_blocks)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, t: Tensor, x: Tensor) -> Tensor:
        # t.shape == (batch_size, n_take)
        # x.shape == (batch_size, n_take, input_size)
        
        # z0.shape == (batch_size, n_blocks, hidden_size)
        z0 = self.h_encoder(x[:, 0, :])

        k = self.transpose_for_scores(torch.matmul(z0, self.key_w_T.T) + self.key_b.expand(self.n_blocks * self.n_blocks, -1).T)
        q = self.transpose_for_scores(torch.matmul(z0, self.query_w) + self.query_b.expand(self.n_blocks * self.n_blocks, -1).T)

        # k.shape, q.shape == (batch_size, 1, n_blocks, n_blocks * n_blocks)
        l0 = torch.matmul(q, k.transpose(-1, -2)).squeeze(dim=1) / self.n_blocks # l0.shape == (batch_size, n_blocks, n_blocks)

        x = self.x_encoder(x) # x.shape == (batch_size, n_blocks, n_take, 2 * input_size)

        temp_t = t.expand(self.n_blocks, -1, -1).permute(1, 0, 2).reshape(-1, t.size(1))
        x_spline = NaturalCubicSpline(temp_t, cubic_spline(temp_t, x.reshape(-1, t.size(1), x.size(-1))))
        temp_t = None

        insert_t = torch.linspace(int(t[0, 0]), int(t[0, -1]), 3 * int(t[0, -1] - t[0, 0]) + 1, device=t.device)

        for i in insert_t:
            a0 = nn.Softmax(dim=-1)(l0)

            # dXdt.shape == (batch_size, n_blocks, 2 * input_size)
            dXdt = x_spline.derivative(i).reshape(t.size(0), self.n_blocks, -1)

            # torch.bmm(a0, z0).shape == (batch_size, n_blocks, hidden_size)
            # self.hidden_func(torch.bmm(a0, z0)).shape == (batch_size, n_blocks, hidden_size, 2 * input_size)
            # dz.shape == (batch_size, n_blocks, hidden_size)

            dz = torch.matmul(
                self.hidden_func(torch.bmm(a0, z0)), # shape == (batch_size, n_blocks, hidden_size, 2 * input_size)
                dXdt.unsqueeze(dim=-1)
            ).squeeze(dim=-1)

            z0 = z0 + dz / 3

            ##############################################################
            #####          n is n_blocks, h is hidden _size          #####
            #####    Z: n x h,    K, Q: h x n^2,    k, q: n x n^2    #####
            ##############################################################
            #####              l = (ZQ + q)(ZK + k)^T                #####
            #####    \frac{dl}{dt} = Z' Q K^T Z^T + Z Q K^T Z^{'T}   #####
            #####                        + Z' Q k^T + q K^T Z^{'T}   #####
            ##############################################################

            # dl.shape == (batch_size, n_blocks, n_blocks)
            dl = torch.bmm(torch.matmul(torch.matmul(dz, self.query_w), self.key_w_T), z0.permute(0, 2, 1))

            dl = dl + torch.bmm(torch.matmul(torch.matmul(z0, self.query_w), self.key_w_T), dz.permute(0, 2, 1))
            dl = dl + torch.matmul(torch.matmul(dz, self.query_w), self.key_b.expand(self.n_blocks * self.n_blocks, -1))
            dl = dl + torch.matmul(torch.matmul(self.query_b.expand(self.n_blocks * self.n_blocks, -1).T, self.key_w_T), dz.permute(0, 2, 1))

            l0 = l0 + dl / self.n_blocks / 3

        return self.h_decoder(z0.reshape(-1, self.n_blocks * self.hidden_size))

class SparseDNS(nn.Module):
    def __init__(self, input_size: int = 28, hidden_size: int = 16, output_size: int = 10, n_blocks: int = 6, num_layers: int = 2) -> None:
        super(SparseDNS, self).__init__()
        self.hidden_size = hidden_size
        self.n_blocks = n_blocks
        
        self.x_encoder = GroupLinear_n_to_m(input_size, 2 * input_size, n_blocks)

        self.h_encoder = GroupLinear_1_to_n(input_size, hidden_size, n_blocks)
        self.h_decoder = nn.Linear(n_blocks * hidden_size, output_size)

        self.hidden_func = CDEFunc(2 * input_size, hidden_size, num_layers)

        self.key_w_T = nn.Parameter(0.01 * torch.randn(n_blocks * n_blocks, hidden_size))
        self.query_w = nn.Parameter(0.01 * torch.randn(hidden_size, n_blocks * n_blocks))

        self.key_b = nn.Parameter(torch.zeros(n_blocks))
        self.query_b = nn.Parameter(torch.zeros(n_blocks))

    def transpose_for_scores(self, x: Tensor) -> Tensor:
        new_x_shape = x.size()[:-1] + (1, self.n_blocks * self.n_blocks)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, t: Tensor, x: Tensor) -> Tensor:
        # t.shape == (batch_size, n_take)
        # x.shape == (batch_size, n_take, input_size)
        
        # z0.shape == (batch_size, n_blocks, hidden_size)
        z0 = self.h_encoder(x[:, 0, :])

        k = self.transpose_for_scores(torch.matmul(z0, self.key_w_T.T) + self.key_b.expand(self.n_blocks * self.n_blocks, -1).T)
        q = self.transpose_for_scores(torch.matmul(z0, self.query_w) + self.query_b.expand(self.n_blocks * self.n_blocks, -1).T)

        # k.shape, q.shape == (batch_size, 1, n_blocks, n_blocks * n_blocks)
        l0 = torch.matmul(q, k.transpose(-1, -2)).squeeze(dim=1) / self.n_blocks # l0.shape == (batch_size, n_blocks, n_blocks)

        x = self.x_encoder(x) # x.shape == (batch_size, n_blocks, n_take, 2 * input_size)

        temp_t = t.expand(self.n_blocks, -1, -1).permute(1, 0, 2).reshape(-1, t.size(1))
        x_spline = NaturalCubicSpline(temp_t, cubic_spline(temp_t, x.reshape(-1, t.size(1), x.size(-1))))
        temp_t = None

        insert_t = torch.linspace(int(t[0, 0]), int(t[0, -1]), 3 * int(t[0, -1] - t[0, 0]) + 1, device=t.device)

        for i in insert_t:
            a0 = sparsemax(l0, -1)

            # dXdt.shape == (batch_size, n_blocks, 2 * input_size)
            dXdt = x_spline.derivative(i).reshape(t.size(0), self.n_blocks, -1)

            # torch.bmm(a0, z0).shape == (batch_size, n_blocks, hidden_size)
            # self.hidden_func(torch.bmm(a0, z0)).shape == (batch_size, n_blocks, hidden_size, 2 * input_size)
            # dz.shape == (batch_size, n_blocks, hidden_size)

            dz = torch.matmul(
                self.hidden_func(torch.bmm(a0, z0)), # shape == (batch_size, n_blocks, hidden_size, 2 * input_size)
                dXdt.unsqueeze(dim=-1)
            ).squeeze(dim=-1)

            z0 = z0 + dz / 3

            ##############################################################
            #####          n is n_blocks, h is hidden _size          #####
            #####    Z: n x h,    K, Q: h x n^2,    k, q: n x n^2    #####
            ##############################################################
            #####              l = (ZQ + q)(ZK + k)^T                #####
            #####    \frac{dl}{dt} = Z' Q K^T Z^T + Z Q K^T Z^{'T}   #####
            #####                        + Z' Q k^T + q K^T Z^{'T}   #####
            ##############################################################

            # dl.shape == (batch_size, n_blocks, n_blocks)
            dl = torch.bmm(torch.matmul(torch.matmul(dz, self.query_w), self.key_w_T), z0.permute(0, 2, 1))

            dl = dl + torch.bmm(torch.matmul(torch.matmul(z0, self.query_w), self.key_w_T), dz.permute(0, 2, 1))
            dl = dl + torch.matmul(torch.matmul(dz, self.query_w), self.key_b.expand(self.n_blocks * self.n_blocks, -1))
            dl = dl + torch.matmul(torch.matmul(self.query_b.expand(self.n_blocks * self.n_blocks, -1).T, self.key_w_T), dz.permute(0, 2, 1))

            l0 = l0 + dl / self.n_blocks / 3

        return self.h_decoder(z0.reshape(-1, self.n_blocks * self.hidden_size))

class DisDNS(nn.Module):
    def __init__(self, input_size: int = 28, hidden_size: int = 16, output_size: int = 10, n_blocks: int = 6, num_layers: int = 2) -> None:
        super(DisDNS, self).__init__()
        self.hidden_size = hidden_size
        self.n_blocks = n_blocks
        
        self.x_encoder = GroupLinear_n_to_m(input_size, 2 * input_size, n_blocks)

        self.h_encoder = GroupLinear_1_to_n(input_size, hidden_size, n_blocks)
        self.h_decoder = nn.Linear(n_blocks * hidden_size, output_size)

        self.hidden_func = CDEFunc(2 * input_size, hidden_size, num_layers)

        self.key = nn.Linear(hidden_size, n_blocks * n_blocks)
        self.query = nn.Linear(hidden_size, n_blocks * n_blocks)

    def transpose_for_scores(self, x: Tensor) -> Tensor:
        new_x_shape = x.size()[:-1] + (1, self.n_blocks * self.n_blocks)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, t: Tensor, x: Tensor) -> Tensor:
        # t.shape == (batch_size, n_take)
        # x.shape == (batch_size, n_take, input_size)
        
        # z0.shape == (batch_size, n_blocks, hidden_size)
        z0 = self.h_encoder(x[:, 0, :])

        x = self.x_encoder(x) # x.shape == (batch_size, n_blocks, n_take, 2 * input_size)

        temp_t = t.expand(self.n_blocks, -1, -1).permute(1, 0, 2).reshape(-1, t.size(1))
        x_spline = NaturalCubicSpline(temp_t, cubic_spline(temp_t, x.reshape(-1, t.size(1), x.size(-1))))
        temp_t = None

        insert_t = torch.linspace(int(t[0, 0]), int(t[0, -1]), 3 * int(t[0, -1] - t[0, 0]) + 1, device=t.device)

        for i in insert_t:
            k = self.transpose_for_scores(self.key(z0))
            q = self.transpose_for_scores(self.query(z0))

            # k.shape, q.shape == (batch_size, 1, n_blocks, n_blocks * n_blocks)
            a0 = nn.Softmax(dim=-1)(torch.matmul(q, k.transpose(-1, -2)).squeeze(dim=1) / self.n_blocks)

            # dXdt.shape == (batch_size, n_blocks, 2 * input_size)
            dXdt = x_spline.derivative(i).reshape(t.size(0), self.n_blocks, -1)

            # torch.bmm(a0, z0).shape == (batch_size, n_blocks, hidden_size)
            # self.hidden_func(torch.bmm(a0, z0)).shape == (batch_size, n_blocks, hidden_size, 2 * input_size)
            # dz.shape == (batch_size, n_blocks, hidden_size)

            dz = torch.matmul(
                self.hidden_func(torch.bmm(a0, z0)), # shape == (batch_size, n_blocks, hidden_size, 2 * input_size)
                dXdt.unsqueeze(dim=-1)
            ).squeeze(dim=-1)

            z0 = z0 + dz / 3

        return self.h_decoder(z0.reshape(-1, self.n_blocks * self.hidden_size))

class NeuralCDE(nn.Module):
    def __init__(self, input_size: int = 28, hidden_size: int = 16, output_size: int = 10, num_layers: int = 2) -> None:
        super(NeuralCDE, self).__init__()
        self.hidden_size = hidden_size
        
        self.h_encoder = nn.Linear(input_size, hidden_size)
        self.h_decoder = nn.Linear(hidden_size, output_size)

        self.hidden_func = CDEFunc(input_size, hidden_size, num_layers)

    def forward(self, t: Tensor, x: Tensor) -> Tensor:
        # t.shape == (batch_size, n_take)
        # x.shape == (batch_size, n_take, input_size)
        z0 = self.h_encoder(x[:, 0, :])

        x_spline = NaturalCubicSpline(t, cubic_spline(t, x))

        insert_t = torch.linspace(int(t[0, 0]), int(t[0, -1]), 3 * int(t[0, -1] - t[0, 0]) + 1, device=t.device)

        for i in insert_t:
            dz = torch.bmm(
                self.hidden_func(z0), # shape == (batch_size, n_blocks, hidden_size, 2 * input_size)
                x_spline.derivative(i).unsqueeze(dim=-1)
            ).squeeze(dim=-1)

            z0 = z0 + dz / 3

        return self.h_decoder(z0)

def tridiagonal_solve(b_: Tensor, A_upper_: Tensor, A_diagonal_: Tensor, A_lower_: Tensor) -> Tensor:
    """Solves a tridiagonal system Ax = b.

    The arguments A_upper, A_digonal, A_lower correspond to the three diagonals of A. Letting U = A_upper, D=A_digonal
    and L = A_lower, and assuming for simplicity that there are no batch dimensions, then the matrix A is assumed to be
    of size (k, k), with entries:

    D[0] U[0]
    L[0] D[1] U[1]
         L[1] D[2] U[2]                     0
              L[2] D[3] U[3]
                  .    .    .
                       .      .      .
                           .        .        .
                        L[k - 3] D[k - 2] U[k - 2]
           0                     L[k - 2] D[k - 1] U[k - 1]
                                          L[k - 1]   D[k]

    Arguments:
        b: A tensor of shape (..., k), where '...' is zero or more batch dimensions
        A_upper: A tensor of shape (..., k - 1).
        A_diagonal: A tensor of shape (..., k).
        A_lower: A tensor of shape (..., k - 1).

    Returns:
        A tensor of shape (..., k), corresponding to the x solving Ax = b

    Warning:
        This implementation isn't super fast. You probably want to cache the result, if possible.
    """

    # This implementation is very much written for clarity rather than speed.

    A_upper = torch.empty(b_.size(0), b_.size(1), b_.size(2) - 1, dtype=b_.dtype, device=b_.device)
    A_lower = torch.empty(b_.size(0), b_.size(1), b_.size(2) - 1, dtype=b_.dtype, device=b_.device)
    A_diagonal = torch.empty(*b_.shape, dtype=b_.dtype, device=b_.device)
    b = torch.empty(*b_.shape, dtype=b_.dtype, device=b_.device)

    for i in range(b_.size(0)):
        A_upper[i], _ = torch.broadcast_tensors(A_upper_[i], b_[i, :, :-1])
        A_lower[i], _ = torch.broadcast_tensors(A_lower_[i], b_[i, :, :-1])
        A_diagonal[i], b[i] = torch.broadcast_tensors(A_diagonal_[i], b_[i])

    channels = b.size(-1)

    new_shape = (b.size(0), channels, b.size(1))
    new_b = torch.zeros(*new_shape, dtype=b.dtype, device=b_.device)
    new_A_diagonal = torch.empty(*new_shape, dtype=b.dtype, device=b_.device)
    outs = torch.empty(*new_shape, dtype=b.dtype, device=b_.device)
    
    new_b[:, 0] = b[..., 0]
    new_A_diagonal[:, 0] = A_diagonal[..., 0]
    for i in range(1, channels):
        w = A_lower[..., i - 1] / new_A_diagonal[:, i - 1]
        new_A_diagonal[:, i] = A_diagonal[..., i] - w * A_upper[..., i - 1]
        new_b[:, i] = b[..., i] - w * new_b[:, i - 1]

    outs[:, channels - 1] = new_b[:, channels - 1] / new_A_diagonal[:, channels - 1]
    for i in range(channels - 2, -1, -1):
        outs[:, i] = (new_b[:, i] - A_upper[..., i] * outs[:, i + 1]) / new_A_diagonal[:, i]

    return outs.permute(0, 2, 1)

def cubic_spline(times: Tensor, x: Tensor) -> List[Tensor]:
    path = x.transpose(-1, -2)
    length = path.size(-1)

    # Set up some intermediate values
    time_diffs = times[:, 1:] - times[:, :-1]
    time_diffs_reciprocal = time_diffs.reciprocal()
    time_diffs_reciprocal_squared = time_diffs_reciprocal ** 2

    three_path_diffs = 3 * (path[..., 1:] - path[..., :-1])
    six_path_diffs = 2 * three_path_diffs

    # path_diffs_scaled.shape == (batch_size, input_size, n_take)
    path_diffs_scaled = three_path_diffs * time_diffs_reciprocal_squared.unsqueeze(dim=1)

    # Solve a tridiagonal linear system to find the derivatives at the knots
    system_diagonal = torch.empty(times.size(0), length, dtype=path.dtype, device=path.device)
    system_diagonal[:, :-1] = time_diffs_reciprocal
    system_diagonal[:, -1] = 0
    system_diagonal[:, 1:] += time_diffs_reciprocal
    system_diagonal *= 2
    system_rhs = torch.empty(*path.shape, dtype=path.dtype, device=path.device)
    system_rhs[..., :-1] = path_diffs_scaled
    system_rhs[..., -1] = 0
    system_rhs[..., 1:] += path_diffs_scaled

    knot_derivatives = tridiagonal_solve(system_rhs, time_diffs_reciprocal, system_diagonal, time_diffs_reciprocal)

    a = path[..., :-1]
    b = knot_derivatives[..., :-1]
    two_c = (six_path_diffs * time_diffs_reciprocal.unsqueeze(dim=1)
            - 4 * knot_derivatives[..., :-1]
            - 2 * knot_derivatives[..., 1:]) * time_diffs_reciprocal.unsqueeze(dim=1)
    three_d = (-six_path_diffs * time_diffs_reciprocal.unsqueeze(dim=1)
            + 3 * (knot_derivatives[..., :-1]
                    + knot_derivatives[..., 1:])) * time_diffs_reciprocal_squared.unsqueeze(dim=1)

    return a.transpose(-1, -2), b.transpose(-1, -2), two_c.transpose(-1, -2), three_d.transpose(-1, -2)

class blocked_grad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, mask):
        ctx.save_for_backward(x, mask)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        _, mask = ctx.saved_tensors
        return grad_output * mask, mask * 0.0

class GroupLinearLayer(nn.Module):
    def __init__(self, din: int, dout: int, num_blocks: int) -> None:
        super(GroupLinearLayer, self).__init__()

        self.w = nn.Parameter(0.01 * torch.randn(num_blocks, din, dout))

    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(1, 0, 2)
        x = torch.bmm(x, self.w)
        return x.permute(1,0,2)

class GroupLSTMCell(nn.Module):
    """
    GroupLSTMCell can compute the operation of N LSTM Cells at once.
    """
    def __init__(self, inp_size: int, hidden_size: int, num_lstms: int) -> None:
        super().__init__()
        self.inp_size = inp_size
        self.hidden_size = hidden_size
        
        self.i2h = GroupLinearLayer(inp_size, 4 * hidden_size, num_lstms)
        self.h2h = GroupLinearLayer(hidden_size, 4 * hidden_size, num_lstms)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        stdv = 1.0 / np.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x: Tensor, hid_state: Tensor) -> Tensor:
        """
        input: x (batch_size, num_lstms, input_size)
            hid_state (tuple of length 2 with each element of size (batch_size, num_lstms, hidden_state))
        output: h (batch_size, num_lstms, hidden_state)
                c ((batch_size, num_lstms, hidden_state))
        """
        h, c = hid_state

        preact = self.i2h(x) + self.h2h(h)

        gates = preact[:, :,  :3 * self.hidden_size].sigmoid()
        
        g_t = preact[:, :,  3 * self.hidden_size:].tanh()
        
        i_t = gates[:, :,  :self.hidden_size]

        f_t = gates[:, :, self.hidden_size:2 * self.hidden_size]

        o_t = gates[:, :, -self.hidden_size:]

        c_t = torch.mul(c, f_t) + torch.mul(i_t, g_t) 
        h_t = torch.mul(o_t, c_t.tanh())

        return h_t, c_t

class RIMCell(nn.Module):
    def __init__(self, 
        device, input_size, hidden_size, num_units, k, input_key_size = 64, input_value_size = 400, input_query_size = 64,
        num_input_heads = 1, input_dropout = 0.1, comm_key_size = 32, comm_value_size = 100, comm_query_size = 32, num_comm_heads = 4, comm_dropout = 0.1
    ) -> None:
        super().__init__()
        if comm_value_size != hidden_size:
            #print('INFO: Changing communication value size to match hidden_size')
            comm_value_size = hidden_size
        self.device = device
        self.hidden_size = hidden_size
        self.num_units =num_units
        self.key_size = input_key_size
        self.k = k
        self.num_input_heads = num_input_heads
        self.num_comm_heads = num_comm_heads
        self.input_key_size = input_key_size
        self.input_query_size = input_query_size
        self.input_value_size = input_value_size

        self.comm_key_size = comm_key_size
        self.comm_query_size = comm_query_size
        self.comm_value_size = comm_value_size

        self.key = nn.Linear(input_size, num_input_heads * input_query_size).to(self.device)
        self.value = nn.Linear(input_size, num_input_heads * input_value_size).to(self.device)

        self.rnn = GroupLSTMCell(input_value_size, hidden_size, num_units)
        self.query = GroupLinearLayer(hidden_size,  input_key_size * num_input_heads, self.num_units)
        self.query_ = GroupLinearLayer(hidden_size, comm_query_size * num_comm_heads, self.num_units) 
        self.key_ = GroupLinearLayer(hidden_size, comm_key_size * num_comm_heads, self.num_units)
        self.value_ = GroupLinearLayer(hidden_size, comm_value_size * num_comm_heads, self.num_units)
        self.comm_attention_output = GroupLinearLayer(num_comm_heads * comm_value_size, comm_value_size, self.num_units)
        self.comm_dropout = nn.Dropout(p =input_dropout)
        self.input_dropout = nn.Dropout(p =comm_dropout)


    def transpose_for_scores(self, x, num_attention_heads, attention_head_size):
        new_x_shape = x.size()[:-1] + (num_attention_heads, attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def input_attention_mask(self, x, h):
        """
        Input : x (batch_size, 2, input_size) [The null input is appended along the first dimension]
                h (batch_size, num_units, hidden_size)
        Output: inputs (list of size num_units with each element of shape (batch_size, input_value_size))
                mask_ binary array of shape (batch_size, num_units) where 1 indicates active and 0 indicates inactive
        """
        # self.key(x), self.value(x): Class nn.Linear
        # self.query(x): Class GroupLinearLayer
        key_layer = self.key(x)
        value_layer = self.value(x)
        query_layer = self.query(h)

        key_layer = self.transpose_for_scores(key_layer, self.num_input_heads, self.input_key_size)
        value_layer = torch.mean(self.transpose_for_scores(value_layer, self.num_input_heads, self.input_value_size), dim = 1)
        query_layer = self.transpose_for_scores(query_layer, self.num_input_heads, self.input_query_size)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) / np.sqrt(self.input_key_size) 
        attention_scores = torch.mean(attention_scores, dim = 1)

        mask_ = torch.zeros(x.size(0), self.num_units).to(self.device)

        not_null_scores = attention_scores[:,:, 0]
        # torch.topk(input, k, dim=None, largest=True, sorted=True, *, out=None)
        # Returns the k largest elements of the given input tensor along a given dimension.
        topk1 = torch.topk(not_null_scores, self.k, dim = 1)
        row_index = np.arange(x.size(0))
        row_index = np.repeat(row_index, self.k)

        # Pairwisely assign 1
        mask_[row_index, topk1.indices.view(-1)] = 1
        
        attention_probs = self.input_dropout(nn.Softmax(dim = -1)(attention_scores))
        inputs = torch.matmul(attention_probs, value_layer) * mask_.unsqueeze(2)

        return inputs, mask_

    def communication_attention(self, h, mask):
        """
        Input : h (batch_size, num_units, hidden_size)
                mask obtained from the input_attention_mask() function
        Output: context_layer (batch_size, num_units, hidden_size). New hidden states after communication
        """
        query_layer = []
        key_layer = []
        value_layer = []
        
        query_layer = self.query_(h)
        key_layer = self.key_(h)
        value_layer = self.value_(h)

        query_layer = self.transpose_for_scores(query_layer, self.num_comm_heads, self.comm_query_size)
        key_layer = self.transpose_for_scores(key_layer, self.num_comm_heads, self.comm_key_size)
        value_layer = self.transpose_for_scores(value_layer, self.num_comm_heads, self.comm_value_size)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / np.sqrt(self.comm_key_size)
        
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        
        mask = [mask for _ in range(attention_probs.size(1))]
        # torch.stack(tensors, dim=0, *, out=None) → Tensor
        # Concatenates a sequence of tensors along a new dimension.
        # All tensors need to be of the same size.
        mask = torch.stack(mask, dim = 1)
        
        attention_probs = attention_probs * mask.unsqueeze(3)
        attention_probs = self.comm_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.num_comm_heads * self.comm_value_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        context_layer = self.comm_attention_output(context_layer)
        context_layer = context_layer + h
        
        return context_layer

    def forward(self, x, hs, cs = None):
        """
        Input : x (batch_size, 1 , input_size)
                hs (batch_size, num_units, hidden_size)
                cs (batch_size, num_units, hidden_size)
        Output: new hs, cs for LSTM
                new hs for GRU
        """
        size = x.size()
        null_input = torch.zeros(size[0], 1, size[2]).float().to(self.device)
        x = torch.cat((x, null_input), dim = 1)

        # Compute input attention
        inputs, mask = self.input_attention_mask(x, hs)
        h_old = hs * 1.0
        if cs is not None:
            c_old = cs * 1.0
        

        # Compute RNN(LSTM or GRU) output
        
        if cs is not None:
            hs, cs = self.rnn(inputs, (hs, cs))
        else:
            hs = self.rnn(inputs, hs)

        # Block gradient through inactive units
        mask = mask.unsqueeze(2)
        h_new = blocked_grad.apply(hs, mask)

        # Compute communication attention
        h_new = self.communication_attention(h_new, mask.squeeze(2))

        hs = mask * h_new + (1 - mask) * h_old
        if cs is not None:
            cs = mask * cs + (1 - mask) * c_old
            return hs, cs

        return hs, None

class RIM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, n_blocks: int = 6, n_update: int = 3) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.n_blocks = n_blocks

        self.rim_model = RIMCell(DEVICE, input_size, hidden_size, n_blocks, n_update, 64, 400, 64,
            1, 0.1, 32, 100, 32, 1, 0.1).to(DEVICE)

        self.linear = nn.Linear(hidden_size * n_blocks, output_size)

    def forward(self, t: Tensor, x: Tensor) -> Tensor:
        hs = torch.randn((x.size(0), self.n_blocks, self.hidden_size), device=x.device)

        cs = torch.randn((x.size(0), self.n_blocks, self.hidden_size), device=x.device)

        xs = torch.split(x, 1, 1)

        for x in xs:
            hs, cs = self.rim_model(x, hs, cs)

        return self.linear(hs.contiguous().reshape(-1, self.n_blocks * self.hidden_size))