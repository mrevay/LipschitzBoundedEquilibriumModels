import torch
import torch.nn as nn
import torch.nn.functional as F


class lben_fc_block(nn.Module):

    def __init__(self, in_dim, neurons, out_dim, gamma, m=1E-4, epsilon=1E-4):
        super().__init__()

        self.nu = in_dim
        self.nx = neurons
        self.ny = out_dim
        self.gamma = gamma

        self.U = nn.Linear(in_dim, neurons)
        self.V = nn.Linear(neurons, neurons, bias=False)
        self.S = nn.Linear(neurons, neurons, bias=False)

        self.C = nn.Linear(neurons, out_dim)
        self.D = nn.Linear(in_dim, out_dim)

        self.psi = nn.Parameter(torch.zeros((out_dim)))
        self.m = m
        self.epsilon = epsilon


class multi_scale_conv_block(nn.Module):
    def __init__(self, in_channels, channel_dims, out_channels, strides, kernel_sizes):

        padding = 4

        # Input mappings for U
        self.U = nn.ModuleList(nn.Conv2d(in_channels, ch_i, ks_i, str_i) for (
            ch_i, ks_i, str_i) in zip(channel_dims, kernel_sizes, strides))

        self.V = nn.ModuleList(nn.Conv2d(ch_i, ch_i, ks_i, str_i) for (
            ch_i, ks_i, str_i) in zip(channel_dims, kernel_sizes, strides))

        self.S = nn.ModuleList(nn.Conv2d(ch_i, ch_i, ks_i, str_i) for (
            ch_i, ks_i, str_i) in zip(channel_dims, kernel_sizes, strides))

        self.C = nn.ModuleList(nn.Conv2d(ch_i, out_channels, ks_i, str_i) for (
            ch_i, ks_i, str_i) in zip(channel_dims, kernel_sizes, strides))

        self.D = nn.ModuleList(nn.Conv2d(ch_i, out_channels, ks_i, str_i) for (
            ch_i, ks_i, str_i) in zip(channel_dims, kernel_sizes, strides))


class multi_scale_sequential_conv_block(nn.Module):
    def __init__(self, in_channels, channel_dims, out_channels, strides, kernel_sizes):
        self.V = nn.ModuleList
