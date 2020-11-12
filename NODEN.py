import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import time

import matplotlib.pyplot as plt


def view_conv(z, rows, cols, index):
    for b in range(rows*cols):
        plt.subplot(rows, cols, b+1)
        plt.imshow(z[index, b, :, :].detach().cpu().numpy())

    plt.show()


class Lipschitz_mon(nn.Module):
    """ Simple MON linear class, just a single full multiply. """

    def __init__(self, in_dim, width, out_dim, gamma, m=1.0):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.B = nn.Linear(in_dim, width)
        self.V = nn.Linear(width, width, bias=False)
        self.S = nn.Linear(width, width, bias=False)

        self.Lambda = torch.ones((width))

        self.G = torch.nn.Linear(width, out_dim)

        self.gamma = gamma
        self.m = m
        self.epsilon = 1E-5

    def x_shape(self, n_batch):
        return (n_batch, self.U.in_features)

    def z_shape(self, n_batch):
        return ((n_batch, self.V.in_features),)

    def forward(self, x, *z):
        return (self.B(x) + self.multiply(*z)[0],)

    def bias(self, x):
        return (self.B(x),)

    def multiply(self, *z):

        z_out = z[0] @ self.W().T
        return (z_out,)

    def multiply_transpose(self, *g):
        g_out = g[0] @ self.W()
        return (g_out,)

    def init_inverse(self, alpha, beta):
        Id = torch.eye(self.V.weight.shape[0], dtype=self.V.weight.dtype,
                       device=self.V.weight.device)

        W = self.W()
        self.Winv = torch.inverse(alpha * Id + beta * W)

    def inverse(self, *z):
        return (z[0] @ self.Winv.transpose(0, 1),)

    def inverse_transpose(self, *g):
        return (g[0] @ self.Winv,)

    def W(self):

        Id = torch.eye(self.V.weight.shape[0], dtype=self.V.weight.dtype,
                       device=self.V.weight.device)
        VTV = self.V.weight.T @ self.V.weight + self.m * Id
        S = self.S.weight
        GTG = self.G.weight.T @ self.G.weight
        BBT = self.Lambda.diag() @ self.B.weight @ self.B.weight.T @ self.Lambda.diag()

        D = 2 * self.Lambda.diag() - GTG / self.gamma - BBT / self.gamma - VTV + S.T - S
        Lambdainv = (1 / self.Lambda).diag()
        W = 0.5 * Lambdainv @ D
        return W


class NODEN_Lipschitz_Fc(nn.Module):
    """ Simple MON linear class, just a single full multiply. """

    def __init__(self, in_dim, width, out_dim, gamma, m=1.0):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.B = nn.Linear(in_dim, width)
        self.V = nn.Linear(width, width, bias=False)
        self.S = nn.Linear(width, width, bias=False)

        # self.Lambda = nn.Parameter(torch.ones((width)))
        self.psi = nn.Parameter(torch.ones((width)))

        self.G = torch.nn.Linear(width, out_dim)

        self.gamma = gamma
        self.m = m
        self.epsilon = 1E-5

    def x_shape(self, n_batch):
        return (n_batch, self.U.in_features)

    def z_shape(self, n_batch):
        return ((n_batch, self.V.in_features),)

    def forward(self, x, *z):
        return (self.B(x) + self.multiply(*z)[0],)

    def bias(self, x):
        return (self.B(x),)

    def multiply(self, *z):

        z_out = z[0] @ self.W().T
        return (z_out,)

    def multiply_transpose(self, *g):
        g_out = g[0] @ self.W()
        return (g_out,)

    def init_inverse(self, alpha, beta):
        Id = torch.eye(self.V.weight.shape[0], dtype=self.V.weight.dtype,
                       device=self.V.weight.device)

        W = self.W()
        self.Winv = torch.inverse(alpha * Id + beta * W)

    def inverse(self, *z):
        return (z[0] @ self.Winv.transpose(0, 1),)

    def inverse_transpose(self, *g):
        return (g[0] @ self.Winv,)

    def W(self):

        Psi = torch.exp(self.psi)

        Id = torch.eye(self.V.weight.shape[0], dtype=self.V.weight.dtype,
                       device=self.V.weight.device)

        VTV = self.V.weight.T @ self.V.weight + self.m * Id
        S = self.S.weight
        GTG = self.G.weight.T @ self.G.weight

        # BBT = self.Lambda.diag() @ self.B.weight @ self.B.weight.T @ self.Lambda.diag()
        Psi_inv = (1 / Psi).diag()
        BBT = Psi_inv @ self.B.weight @ self.B.weight.T @ Psi_inv

        W = Id - Psi.diag() @ (GTG / 2 / self.gamma + BBT / 2 / self.gamma + VTV + S.T - S)
        return W


class NODEN_SingleFc(nn.Module):
    """ Simple MON linear class, just a single full multiply. """

    def __init__(self, in_dim, out_dim, m=1.0):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.U = nn.Linear(in_dim, out_dim)
        self.V = nn.Linear(out_dim, out_dim, bias=False)
        self.S = nn.Linear(out_dim, out_dim, bias=False)

        self.psi = nn.Parameter(torch.zeros((out_dim)))
        # self.Lambda = (torch.ones((out_dim)))
        self.m = m
        self.epsilon = 1E-5

    def x_shape(self, n_batch):
        return (n_batch, self.U.in_features)

    def z_shape(self, n_batch):
        return ((n_batch, self.V.in_features),)

    def forward(self, x, *z):
        return (self.U(x) + self.multiply(*z)[0],)

    def bias(self, x):
        return (self.U(x),)

    def multiply(self, *z):

        z_out = z[0] @ self.W().T
        return (z_out,)

    def multiply_transpose(self, *g):
        g_out = g[0] @ self.W()
        return (g_out,)

    def init_inverse(self, alpha, beta):
        Id = torch.eye(self.V.weight.shape[0], dtype=self.V.weight.dtype,
                       device=self.V.weight.device)

        W = self.W()
        self.Winv = torch.inverse(alpha * Id + beta * W)

    def inverse(self, *z):
        return (z[0] @ self.Winv.transpose(0, 1),)

    def inverse_transpose(self, *g):
        return (g[0] @ self.Winv,)

    def W(self):

        Psi = torch.exp(self.psi).diag()
        Id = torch.eye(self.V.weight.shape[0], dtype=self.V.weight.dtype,
                       device=self.V.weight.device)
        VTVz = self.V.weight.T @ self.V.weight + self.m * Id
        S = self.S.weight
        # W = Id - self.Lambda.diag() @ (VTVz + S.T - S)
        W = Id - Psi @ (VTVz + S.T - S)

        return W


class Uncon_Conv(nn.Module):
    """ MON class with a single 3x3 (circular) convolution """

    def __init__(self, in_channels, out_channels, shp, kernel_size=5, m=1.0):
        super().__init__()

        self.pad = 4 * ((kernel_size - 1) // 2,)
        self.shp = shp
        self.m = m
        self.h = nn.Parameter(torch.tensor(1.))

        self.U = nn.Conv2d(in_channels, out_channels, kernel_size)

        # Initialize unconstrained W in the same way as mon and Lode
        self.W = nn.Conv2d(out_channels, out_channels, kernel_size, bias=False)

    def cpad(self, x):
        return F.pad(x, self.pad, mode="circular")

    def uncpad(self, x):
        return x[:, :, 2 * self.pad[0]:-2 * self.pad[1], 2 * self.pad[2]:-2 * self.pad[3]]

    def x_shape(self, n_batch):
        return (n_batch, self.U.in_channels, self.shp[0], self.shp[1])

    def z_shape(self, n_batch):
        return ((n_batch, self.W.in_channels, self.shp[0], self.shp[1]),)

    def forward(self, x, *z):
        # circular padding is broken in PyTorch
        return (F.conv2d(self.cpad(x), self.U.weight, self.U.bias) + self.multiply(*z)[0],)

    def bias(self, x):
        return (F.conv2d(self.cpad(x), self.U.weight, self.U.bias),)

    def multiply(self, *z):
        W = self.h * self.W.weight / self.W.weight.view(-1).norm()
        Wz = F.conv2d(self.cpad(z[0]), W)
        z_out = Wz
        return (z_out,)

    def multiply_transpose(self, *g):
        W = self.h * self.W.weight / self.W.weight.view(-1).norm()
        Wz = self.uncpad(F.conv_transpose2d(self.cpad(g[0]), W))
        g_out = Wz

        return (g_out,)

    def max_sv(self):

        W = self.h * self.W.weight / self.W.weight.view(-1).norm()

        Wfft = init_fft_conv(W, self.shp)
        eye = torch.eye(Wfft.shape[1], dtype=Wfft.dtype,
                        device=Wfft.device)[None, :, :]

        ImW = eye - Wfft

        return ImW.svd()[1].max()

    # def init_inverse(self, alpha, beta):
    #     A = self.A.weight / self.A.weight.view(-1).norm()
    #     B = self.h * self.B.weight / self.B.weight.view(-1).norm()
    #     Afft = init_fft_conv(A, self.shp)
    #     Bfft = init_fft_conv(B, self.shp)
    #     I = torch.eye(Afft.shape[1], dtype=Afft.dtype,
    #                   device=Afft.device)[None, :, :]
    #     self.Wfft = (1 - self.m) * I - self.g * Afft.transpose(1, 2) @ Afft + Bfft - Bfft.transpose(1, 2)
    #     self.Winv = torch.inverse(alpha * I + beta * self.Wfft)

    #     # Store the value of alpha. This is bad code though...
    #     self.alpha = -beta

    def inverse(self, *z):
        return (fft_conv(z[0], self.Winv),)

    def inverse_transpose(self, *g):
        return (fft_conv(g[0], self.Winv, transpose=True),)


class NODEN_SingleFc_uncon(nn.Module):
    """ Simple MON linear class, just a single full multiply. """

    def __init__(self, in_dim, out_dim, m=1.0):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.U = nn.Linear(in_dim, out_dim)
        V = nn.Linear(out_dim, out_dim, bias=False)
        S = nn.Linear(out_dim, out_dim, bias=False)

        Lambda = torch.ones((out_dim))

        Id = torch.eye(V.weight.shape[0], dtype=V.weight.dtype,
                       device=V.weight.device)
        VTVz = V.weight.T @ V.weight + m * Id
        S = S.weight

        self.Wp = torch.nn.Parameter(Id - Lambda.diag() @ (VTVz + S.T - S))

        self.m = m
        self.epsilon = 1E-5

    def x_shape(self, n_batch):
        return (n_batch, self.U.in_features)

    def z_shape(self, n_batch):
        return ((n_batch, self.Wp.shape[1]),)

    def forward(self, x, *z):
        return (self.U(x) + self.multiply(*z)[0],)

    def bias(self, x):
        return (self.U(x),)

    def multiply(self, *z):

        z_out = z[0] @ self.W().T
        return (z_out,)

    def multiply_transpose(self, *g):
        g_out = g[0] @ self.W()

        return (g_out,)

    def init_inverse(self, alpha, beta):
        Id = torch.eye(self.Wp.shape[0], dtype=self.Wp.dtype,
                       device=self.Wp.device)

        W = self.W()

        self.Winv = torch.inverse(alpha * Id + beta * W)

    def inverse(self, *z):
        return (z[0] @ self.Winv.transpose(0, 1),)

    def inverse_transpose(self, *g):
        return (g[0] @ self.Winv,)

    def W(self):
        return self.Wp


class NODEN_Conv(nn.Module):
    """ Simple MON linear class, just a single full multiply. """

    def __init__(self, in_dim, in_channels, out_channels, shp, kernel_size=3, m=1.0):
        super().__init__()
        self.U = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.A = nn.Conv2d(out_channels, out_channels, kernel_size, bias=False)
        self.g = nn.Parameter(torch.tensor(1.))
        self.h = nn.Parameter(torch.tensor(1.))
        self.B = nn.Conv2d(out_channels, out_channels, kernel_size, bias=False)
        self.pad = 4 * ((kernel_size - 1) // 2,)
        self.shp = shp
        self.m = m

        self.psi = nn.Parameter(torch.zeros(
            (1, out_channels, in_dim+2, in_dim+2)))

    def cpad(self, x):
        return F.pad(x, self.pad, mode="circular")

    def uncpad(self, x):
        return x[:, :, 2 * self.pad[0]:-2 * self.pad[1], 2 * self.pad[2]:-2 * self.pad[3]]

    def x_shape(self, n_batch):
        return (n_batch, self.U.in_channels, self.shp[0], self.shp[1])

    def z_shape(self, n_batch):
        return ((n_batch, self.A.in_channels, self.shp[0], self.shp[1]),)

    def forward(self, x, *z):
        # circular padding is broken in PyTorch
        return (F.conv2d(self.cpad(x), self.U.weight, self.U.bias) + self.multiply(*z)[0],)

    def bias(self, x):
        return (F.conv2d(self.cpad(x), self.U.weight, self.U.bias),)

    def multiply(self, *z):
        Psi = torch.exp(self.psi)
        A = self.A.weight / self.A.weight.view(-1).norm()
        B = self.h * self.B.weight / self.B.weight.view(-1).norm()
        Az = F.conv2d(self.cpad(z[0]), A)
        ATAz = self.uncpad(F.conv_transpose2d(self.cpad(Az), A))
        Bz = F.conv2d(self.cpad(z[0]), B)
        BTz = self.uncpad(F.conv_transpose2d(self.cpad(z[0]), B))
        z_out = z[0] - Psi * (self.g * ATAz - Bz + BTz + self.m * z[0])
        return (z_out,)

    def multiply_transpose(self, *g):
        Psi = torch.exp(self.psi)
        gp = Psi * g[0]
        A = self.A.weight / self.A.weight.view(-1).norm()
        B = self.h * self.B.weight / self.B.weight.view(-1).norm()
        Ag = F.conv2d(self.cpad(gp), A)
        ATAg = self.uncpad(F.conv_transpose2d(self.cpad(Ag), A))
        Bg = F.conv2d(self.cpad(gp), B)
        BTg = self.uncpad(F.conv_transpose2d(self.cpad(gp), B))
        g_out = g[0] - (self.g * ATAg + Bg - BTg + self.m * gp)
        return (g_out,)

    def init_inverse(self, alpha, beta):
        # Store the value of alpha. This is bad code though...
        self.alpha = -beta


# class NODEN_Lipschitz_Conv(nn.Module):
#     """ Simple MON linear class, just a single full multiply. """

#     def __init__(self, in_dim, in_channels, out_channels, out_dim, gamma, shp, kernel_size=3, m=0.1, pool=1):
#         super().__init__()
#         self.U = nn.Conv2d(in_channels, out_channels, kernel_size)
#         self.A = nn.Conv2d(out_channels, out_channels, kernel_size, bias=False)

#         self.f = nn.Parameter(torch.tensor(1.))
#         self.g = nn.Parameter(torch.tensor(1.))
#         self.h = nn.Parameter(torch.tensor(1.))

#         self.B = nn.Conv2d(out_channels, out_channels, kernel_size, bias=False)
#         self.pad = 4 * ((kernel_size - 1) // 2,)
#         self.shp = shp
#         self.m = m
#         self.gamma = gamma

#         self.Wout = nn.Linear(out_dim, 10, bias=False)
#         self.pool = pool

#         self.psi = nn.Parameter(torch.zeros(
#             (1, 1, in_dim+2, in_dim+2)))

#     def cpad(self, x):
#         return F.pad(x, self.pad, mode="circular")
#         # return F.pad(x, self.pad, mode='constant')

#     def uncpad(self, x):
#         return x[:, :, 2 * self.pad[0]:-2 * self.pad[1], 2 * self.pad[2]:-2 * self.pad[3]]

#     def x_shape(self, n_batch):
#         return (n_batch, self.U.in_channels, self.shp[0], self.shp[1])

#     def z_shape(self, n_batch):
#         return ((n_batch, self.A.in_channels, self.shp[0], self.shp[1]),)

#     def forward(self, x, *z):
#         # circular padding is broken in PyTorch
#         return (F.conv2d(self.cpad(x), self.U.weight, self.U.bias) + self.multiply(*z)[0],)

#     def bias(self, x):
#         return (F.conv2d(self.cpad(x), self.U.weight, self.U.bias),)

#     def max_sv(self):
#         # A = self.g * self.A.weight / self.A.weight.view(-1).norm()
#         # B = 0*self.h * self.B.weight / self.B.weight.view(-1).norm()
#         # U = self.f * self.U.weight / self.U.weight.view(-1).norm()

#         A = self.A.weight
#         B = self.B.weight
#         U = self.U.weight

#         Afft = init_fft_conv(A, self.shp)
#         Bfft = init_fft_conv(B, self.shp)
#         Ufft = init_fft_conv(U, self.shp)

#         # G = self.Wout.weight.view((-1, A.shape))

#         Psi = torch.exp(self.psi).view((1, Afft.shape[0], 1, 1))
#         # Psi = torch.ones_like(self.psi).view((1, Afft.shape[0], 1, 1))

#         I = torch.eye(Afft.shape[1], dtype=Afft.dtype,
#                       device=Afft.device)[None, :, :]

#         BfftT = Bfft.transpose(1, 2)
#         ATA = Afft.transpose(1, 2) @ Afft
#         UTU = Ufft @ Ufft.transpose(1, 2)

#         # rho(GGT) = rho(GTG).
#         GGT = self.Wout.weight @ self.Wout.weight.T

#         # W except for GTG term.
#         Wfft = I - Psi * (ATA + BfftT - Bfft
#                           + 0.5 * UTU / self.gamma + self.m * I)
#         ImW = I - Wfft

#         # Upper bounds true sv via triangle inequality.
#         return ImW.svd(compute_uv=False)[1].max() + 0.5*torch.svd_lowrank(GGT)[1][0] / self.gamma

#     def output_pooling(self, x):
#         # The pooling operation used for the output.
#         unpadded = self.uncpad(self.cpad(x))
#         return F.avg_pool2d(unpadded, self.pool)

#     def pool_adjoint(self, x):
#         'Calculate the adjoint of average pooling operator'
#         # adj = torch.nn.functional.interpolate(
#         #     x, scale_factor=self.pool) / self.pool**2
#         # return self.cpad(adj)

#         # adj = torch.nn.functional.interpolate(
#         #     x, size=self.shp) / self.pool**2
#         upsample_size = [self.pool * (ni // self.pool) for ni in self.shp]
#         adj = torch.nn.functional.interpolate(
#             x, size=upsample_size) / self.pool**2
#         return self.cpad(adj)

#     def multiply(self, *z):

#         # A = self.g * self.A.weight / self.A.weight.view(-1).norm()
#         # B = 0*self.h * self.B.weight / self.B.weight.view(-1).norm()
#         # U = self.f * self.U.weight / self.U.weight.view(-1).norm()

#         A = self.A.weight
#         B = self.B.weight
#         U = self.U.weight

#         G = self.Wout.weight

#         Az = F.conv2d(self.cpad(z[0]), A)
#         ATAz = self.uncpad(F.conv_transpose2d(
#             self.cpad(Az), A))

#         # Prior code did not calculate correct adjoint.
#         BTz = self.uncpad(self.cpad(F.conv_transpose2d(z[0], B)))
#         Bz = F.conv2d(self.cpad(z[0]), B)

#         Psi = torch.exp(self.psi)
#         # Psi = torch.ones_like(self.psi)

#         # Calculate term Lambda B B^T Lambda z
#         UTz = F.conv_transpose2d(self.cpad(z[0] / Psi), U)
#         UUTz = self.uncpad(F.conv2d(self.cpad(UTz), U)) / Psi

#         # Calculate term G^T G z
#         pz = self.output_pooling(z[0])

#         GTGpz = pz.view(z[0].shape[0], -1) @ G.T @ G
#         pTGTGz = self.pool_adjoint(GTGpz.view_as(pz))

#         z_out = z[0] - Psi*(0.5*pTGTGz / self.gamma + 0.5*UUTz /
#                             self.gamma + ATAz + self.m * z[0] + BTz - Bz)

#         return (z_out,)

#     def multiply_transpose(self, *g):
#         Psi = torch.exp(self.psi)
#         # Psi = torch.ones_like(self.psi)

#         gp = g[0] * Psi

#         A = self.A.weight
#         B = self.B.weight
#         U = self.U.weight

#         # A = self.g * self.A.weight / self.A.weight.view(-1).norm()
#         # B = 0*self.h * self.B.weight / self.B.weight.view(-1).norm()
#         # U = self.f * self.U.weight / self.U.weight.view(-1).norm()
#         G = self.Wout.weight

#         Az = F.conv2d(self.cpad(gp), A)
#         ATAz = self.uncpad(F.conv_transpose2d(self.cpad(Az), A))

#         BTz = self.uncpad(self.cpad(F.conv_transpose2d(gp, B)))
#         Bz = F.conv2d(self.cpad(gp), B)

#         # Calculate term Lambda B B^T Lambda z
#         UTz = F.conv_transpose2d(self.cpad(gp / Psi), U)
#         UUTz = self.uncpad(F.conv2d(self.cpad(UTz), U)) / Psi

#         # Calculate term G^T G z
#         pz = F.avg_pool2d(gp, self.pool)
#         GTGpz = pz.view(gp.shape[0], -1) @ G.T @ G
#         GTGz = self.pool_adjoint(GTGpz.view_as(pz))

#         z_out = g[0] - (0.5*GTGz / self.gamma + 0.5*UUTz /
#                         self.gamma + ATAz + self.m * gp - BTz + Bz)

#         return (z_out,)


class LBENLipConv(nn.Module):
    """ MON class with a single 3x3 (circular) convolution """

    def __init__(self, in_channels, out_channels, out_dim, shp, kernel_size=3, m=1.0, gamma=1.0, pool=4):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.U = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.A = nn.Conv2d(out_channels, out_channels, kernel_size, bias=False)
        self.g = nn.Parameter(torch.tensor(1.))
        self.h = nn.Parameter(torch.tensor(1.))
        self.B = nn.Conv2d(out_channels, out_channels, kernel_size, bias=False)
        self.pad = 4 * ((kernel_size - 1) // 2,)
        self.shp = shp
        self.m = m

        self.psi = nn.Parameter(torch.zeros((out_channels, shp[0], shp[1])))

        self.gamma = gamma
        self.pool = pool

        n = shp[0]
        self.out_dim = out_channels * (n // pool) ** 2
        self.Wout = nn.Linear(self.out_dim, 10)

    def cpad(self, x):
        return F.pad(x, self.pad, mode="circular")

    def uncpad(self, x):
        return x[:, :, 2 * self.pad[0]:-2 * self.pad[1], 2 * self.pad[2]:-2 * self.pad[3]]

    def x_shape(self, n_batch):
        return (n_batch, self.U.in_channels, self.shp[0], self.shp[1])

    def z_shape(self, n_batch):
        return ((n_batch, self.A.in_channels, self.shp[0], self.shp[1]),)

    def forward(self, x, *z):
        # circular padding is broken in PyTorch
        return (F.conv2d(self.cpad(x), self.U.weight, self.U.bias) + self.multiply(*z)[0],)

    def bias(self, x):
        return (F.conv2d(self.cpad(x), self.U.weight, self.U.bias),)

    def pool_adjoint(self, x):
        'Calculate the adjoint of average pooling operator'

        assert self.shp[0] % self.pool == 0, "Pooling factor must be multiple of padded image width"

        adj = torch.nn.functional.interpolate(
            x, size=self.shp) / self.pool**2
        return adj

    def multiply(self, *z):
        A = self.A.weight
        # B = self.h * self.B.weight / self.B.weight.view(-1).norm()
        B = self.B.weight
        U = self.U.weight

        Az = F.conv2d(self.cpad(z[0]), A)
        ATAz = self.uncpad(F.conv_transpose2d(self.cpad(Az), A))
        Bz = F.conv2d(self.cpad(z[0]), B)
        BTz = self.uncpad(F.conv_transpose2d(self.cpad(z[0]), B))

        UTz = F.conv_transpose2d(self.cpad(z[0]), U)
        UUTz = self.uncpad(F.conv2d(self.cpad(UTz), U))

        G = self.Wout.weight
        zp = F.avg_pool2d(z[-1], self.pool)
        zpvec = zp.view(z[0].shape[0], -1)

        pTGTGpz = self.pool_adjoint(((zpvec @ G.T) @ G).view_as(zp))
        # pTGTGz = self.pool_adjoint(GTGz.view_as(zp))

        Psi = torch.exp(self.psi)[None, ...]
        # Psi = torch.ones_like(self.psi)[None, ...]

        z_out = z[0] - Psi * (self.m * z[0] + ATAz - Bz + BTz +
                              0.5 * UUTz / self.gamma +
                              0.5 * pTGTGpz.view_as(z[0]) / self.gamma)

        def inner(x, y):
            return (x*y).sum()

        return (z_out,)

    def multiply_transpose(self, *g):
        # A = self.A.weight / self.A.weight.view(-1).norm()
        A = self.A.weight
        B = self.B.weight
        U = self.U.weight

        Psi = torch.exp(self.psi)[None, ...]
        # Psi = torch.ones_like(self.psi)[None, ...]
        gpsi = g[0] / Psi

        Ag = F.conv2d(self.cpad(gpsi), A)
        ATAg = self.uncpad(F.conv_transpose2d(self.cpad(Ag), A))
        Bg = F.conv2d(self.cpad(gpsi), B)
        BTg = self.uncpad(F.conv_transpose2d(self.cpad(gpsi), B))

        UTz = F.conv_transpose2d(self.cpad(gpsi), U)
        UUTz = self.uncpad(F.conv2d(self.cpad(UTz), U))

        G = self.Wout.weight

        gp = F.avg_pool2d(gpsi, self.pool)
        gpvec = gp.view(gpsi.shape[0], -1)
        # GTGz = (gp @ G.T) @ G

        pTGTGpz = self.pool_adjoint(((gpvec @ G.T) @ G).view_as(gp))

        g_out = g[0] - self.m * gpsi - ATAg - Bg + BTg - \
            0.5 * UUTz / self.gamma - 0.5 * pTGTGpz.view_as(g[0]) / self.gamma

        return (g_out,)


class NODEN_ReLU(nn.Module):
    def forward(self, *z):
        return tuple(F.relu(z_) for z_ in z)

    def derivative(self, *z):
        return tuple((z_ > 0).type_as(z[0]) for z_ in z)


class NDOEN_tanh(nn.Module):
    def forward(self, *z):
        print("Not implemented yet")
        return None

    def derivative(self, *z):
        print("Not implemented yet")
        return None

# Convolutional layers w/ FFT-based inverses


def fft_to_complex_matrix(x):
    """ Create matrix with [a -b; b a] entries for complex numbers. """
    x_stacked = torch.stack((x, torch.flip(x, (4,))),
                            dim=5).permute(2, 3, 0, 4, 1, 5)
    x_stacked[:, :, :, 0, :, 1] *= -1
    return x_stacked.reshape(-1, 2 * x.shape[0], 2 * x.shape[1])


def fft_to_complex_vector(x):
    """ Create stacked vector with [a;b] entries for complex numbers"""
    return x.permute(2, 3, 0, 1, 4).reshape(-1, x.shape[0], x.shape[1] * 2)


def init_fft_conv(weight, hw):
    """ Initialize fft-based convolution.

    Args:
        weight: Pytorch kernel
        hw: (height, width) tuple
    """
    px, py = (weight.shape[2] - 1) // 2, (weight.shape[3] - 1) // 2
    kernel = torch.flip(weight, (2, 3))
    kernel = F.pad(F.pad(kernel, (0, hw[0] - weight.shape[2], 0, hw[1] - weight.shape[3])),
                   (0, py, 0, px), mode="circular")[:, :, py:, px:]
    return fft_to_complex_matrix(torch.rfft(kernel, 2, onesided=False))


def fft_conv(x, w_fft, transpose=False):
    """ Perhaps FFT-based circular convolution.

    Args:
        x: (B, C, H, W) tensor
        w_fft: conv kernel processed by init_fft_conv
        transpose: flag of whether to transpose convolution
    """
    x_fft = fft_to_complex_vector(torch.rfft(x, 2, onesided=False))
    wx_fft = x_fft.bmm(w_fft.transpose(
        1, 2)) if not transpose else x_fft.bmm(w_fft)
    wx_fft = wx_fft.view(x.shape[2], x.shape[3],
                         wx_fft.shape[1], -1, 2).permute(2, 3, 0, 1, 4)
    return torch.irfft(wx_fft, 2, onesided=False)


def ConvSingularValues(kernel, input_shape):
    # transforms = np.fft.fft2(kernel, input_shape, axes=[0, 1])
    transforms = torch.fft(kernel, 2, axes=[0, 1])
    return transforms.svd(compute_uv=False)
