import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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

        self.Lambda = nn.Parameter(torch.ones((width)))

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


class NODEN_SingleFc(nn.Module):
    """ Simple MON linear class, just a single full multiply. """

    def __init__(self, in_dim, out_dim, m=1.0):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.U = nn.Linear(in_dim, out_dim)
        self.V = nn.Linear(out_dim, out_dim, bias=False)
        self.S = nn.Linear(out_dim, out_dim, bias=False)

        self.Lambda = nn.Parameter(torch.ones((out_dim)))
        # self.Lambda = (torch.ones((out_dim)))
        self.m = m
        self.epsilon = 1E-5

    def x_shape(self, n_batch):
        return (n_batch, self.U.in_features)

    def z_shape(self, n_batch):
        return ((n_batch, self.V.in_features),)

    def forward(self, x, *z):
        self.Lambda.data[self.Lambda < self.epsilon] = self.epsilon
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

        Id = torch.eye(self.V.weight.shape[0], dtype=self.V.weight.dtype,
                       device=self.V.weight.device)
        VTVz = self.V.weight.T @ self.V.weight + self.m * Id
        S = self.S.weight
        W = Id - self.Lambda.diag() @ (VTVz + S.T - S)

        return W


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


class NODEN_Lipschitz_Conv(nn.Module):
    """ Simple MON linear class, just a single full multiply. """

    def __init__(self, in_dim, in_channels, out_channels, out_dim, gamma, shp, kernel_size=3, m=0.1, pool=4):
        super().__init__()
        self.U = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.A = nn.Conv2d(out_channels, out_channels, kernel_size, bias=False)
        self.g = nn.Parameter(torch.tensor(1.))
        self.h = nn.Parameter(torch.tensor(1.))
        self.B = nn.Conv2d(out_channels, out_channels, kernel_size, bias=False)
        self.pad = 4 * ((kernel_size - 1) // 2,)
        self.shp = shp
        self.m = m
        self.gamma = gamma

        self.Wout = nn.Linear(out_dim, 10, bias=False)
        self.pool = pool

        self.Psi = nn.Parameter(torch.ones((1, out_channels, in_dim+2, in_dim+2)))

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
        adj = torch.nn.functional.upsample(x, scale_factor=4) / 16
        return self.cpad(adj)

    def multiply(self, *z):

        A = self.A.weight / self.A.weight.view(-1).norm()
        B = self.h * self.B.weight / self.B.weight.view(-1).norm()
        U = self.U.weight / self.U.weight.view(-1).norm()
        G = self.Wout.weight

        Az = F.conv2d(self.cpad(z[0]), A)
        ATAz = self.uncpad(F.conv_transpose2d(self.cpad(Az), A)) + self.m * z[0]
        Bz = F.conv2d(self.cpad(z[0]), B)
        BTz = self.uncpad(F.conv_transpose2d(self.cpad(z[0]), B))

        # Calculate term Lambda B B^T Lambda z
        UTz = F.conv_transpose2d(self.cpad(z[0] / self.Psi), U)
        UUTz = self.uncpad(F.conv2d(self.cpad(UTz), U)) / self.Psi

        # Calculate term G^T G z
        pz = F.avg_pool2d(z[0], self.pool)
        GTGpz = pz.view(z[0].shape[0], -1) @ G.T @ G
        GTGz = self.pool_adjoint(GTGpz.view_as(pz))

        # z_out = (2 * self.Lambda - GTGz / self.gamma - UUTz / self.gamma - ATAz + BTz - Bz) / 2 / self.Lambda
        z_out = z[0] - self.Psi * (GTGz / self.gamma + UUTz / self.gamma + ATAz + BTz - Bz) / 2

        return (z_out,)

    def multiply_transpose(self, *g):
        gp = g[0] * self.Psi

        A = self.A.weight / self.A.weight.view(-1).norm()
        B = self.h * self.B.weight / self.B.weight.view(-1).norm()
        U = self.U.weight / self.U.weight.view(-1).norm()
        G = self.Wout.weight

        Az = F.conv2d(self.cpad(gp), A)
        ATAz = self.uncpad(F.conv_transpose2d(self.cpad(Az), A)) + self.m * gp
        Bz = F.conv2d(self.cpad(gp), B)
        BTz = self.uncpad(F.conv_transpose2d(self.cpad(gp), B))

        # Calculate term Lambda B B^T Lambda z
        UTz = F.conv_transpose2d(self.cpad(gp / self.Psi), U)
        UUTz = self.uncpad(F.conv2d(self.cpad(UTz), U)) / self.Psi

        # Calculate term G^T G z
        pz = F.avg_pool2d(gp, self.pool)
        GTGpz = pz.view(gp.shape[0], -1) @ G.T @ G
        GTGz = self.pool_adjoint(GTGpz.view_as(pz))

        # z_out = (2 * self.Lambda - GTGz / self.gamma - UUTz / self.gamma - ATAz + BTz - Bz) / 2 / self.Lambda
        z_out = g[0] - (GTGz / self.gamma + UUTz / self.gamma + ATAz - BTz + Bz) / 2

        return (z_out,)

    def init_inverse(self, alpha, beta):
        # A = self.A.weight / self.A.weight.view(-1).norm()
        # B = self.h * self.B.weight / self.B.weight.view(-1).norm()
        # Afft = init_fft_conv(A, self.shp)
        # Bfft = init_fft_conv(B, self.shp)
        # I = torch.eye(Afft.shape[1], dtype=Afft.dtype,
        #               device=Afft.device)[None, :, :]
        # self.Wfft = (1 - self.m) * I - self.g * Afft.transpose(1, 2) @ Afft + Bfft - Bfft.transpose(1, 2)
        # self.Winv = torch.inverse(alpha * I + beta * self.Wfft)

        # Store the value of alpha. This is bad code though...
        self.alpha = -beta


    # def inverse_transpose(self, *g):
    #     return (fft_conv(g[0], self.Winv, transpose=True),)

    # def inverse(self, *z):
    #     alpha = self.alpha
    #     with torch.no_grad():
    #         A = self.A.weight / self.A.weight.view(-1).norm()
    #         B = self.h * self.B.weight / self.B.weight.view(-1).norm()

    #         ztotal = z[0]
    #         zn = z[0]
    #         for n in range(200):
    #             Az = F.conv2d(self.cpad(zn), A)
    #             ATAz = self.uncpad(F.conv_transpose2d(self.cpad(Az), A))
    #             Bz = F.conv2d(self.cpad(zn), B)
    #             BTz = self.uncpad(F.conv_transpose2d(self.cpad(zn), B))
    #             zn = -alpha * (self.m * zn + self.g*ATAz - Bz + BTz)

    #             ztotal += zn
    #             if zn.norm() <= 0.01:
    #                 # print(n, 'iterations')
    #                 break

    #     return ztotal


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

        self.Psi = nn.Parameter(torch.ones((1, out_channels, in_dim+2, in_dim+2)))

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
        A = self.A.weight / self.A.weight.view(-1).norm()
        B = self.h * self.B.weight / self.B.weight.view(-1).norm()
        Az = F.conv2d(self.cpad(z[0]), A)
        ATAz = self.uncpad(F.conv_transpose2d(self.cpad(Az), A))
        Bz = F.conv2d(self.cpad(z[0]), B)
        BTz = self.uncpad(F.conv_transpose2d(self.cpad(z[0]), B))
        z_out = z[0] - self.Psi * (self.g * ATAz - Bz + BTz + self.m * z[0])
        return (z_out,)

    def multiply_transpose(self, *g):
        gp = self.Psi * g[0]
        A = self.A.weight / self.A.weight.view(-1).norm()
        B = self.h * self.B.weight / self.B.weight.view(-1).norm()
        Ag = F.conv2d(self.cpad(gp), A)
        ATAg = self.uncpad(F.conv_transpose2d(self.cpad(Ag), A))
        Bg = F.conv2d(self.cpad(gp), B)
        BTg = self.uncpad(F.conv_transpose2d(self.cpad(gp), B))
        g_out = g[0] - (self.g * ATAg + Bg - BTg + self.m * gp)
        return (g_out,)

    def init_inverse(self, alpha, beta):
        # A = self.A.weight / self.A.weight.view(-1).norm()
        # B = self.h * self.B.weight / self.B.weight.view(-1).norm()
        # Afft = init_fft_conv(A, self.shp)
        # Bfft = init_fft_conv(B, self.shp)
        # I = torch.eye(Afft.shape[1], dtype=Afft.dtype,
        #               device=Afft.device)[None, :, :]
        # self.Wfft = (1 - self.m) * I - self.g * Afft.transpose(1, 2) @ Afft + Bfft - Bfft.transpose(1, 2)
        # self.Winv = torch.inverse(alpha * I + beta * self.Wfft)

        # Store the value of alpha. This is bad code though...
        self.alpha = -beta


    # def inverse_transpose(self, *g):
    #     return (fft_conv(g[0], self.Winv, transpose=True),)

    # def inverse(self, *z):
    #     alpha = self.alpha
    #     with torch.no_grad():
    #         A = self.A.weight / self.A.weight.view(-1).norm()
    #         B = self.h * self.B.weight / self.B.weight.view(-1).norm()

    #         ztotal = z[0]
    #         zn = z[0]
    #         for n in range(200):
    #             Az = F.conv2d(self.cpad(zn), A)
    #             ATAz = self.uncpad(F.conv_transpose2d(self.cpad(Az), A))
    #             Bz = F.conv2d(self.cpad(zn), B)
    #             BTz = self.uncpad(F.conv_transpose2d(self.cpad(zn), B))
    #             zn = -alpha * (self.m * zn + self.g*ATAz - Bz + BTz)

    #             ztotal += zn
    #             if zn.norm() <= 0.01:
    #                 # print(n, 'iterations')
    #                 break

    #     return ztotal


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
    x_stacked = torch.stack((x, torch.flip(x, (4,))), dim=5).permute(2, 3, 0, 4, 1, 5)
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
    wx_fft = x_fft.bmm(w_fft.transpose(1, 2)) if not transpose else x_fft.bmm(w_fft)
    wx_fft = wx_fft.view(x.shape[2], x.shape[3], wx_fft.shape[1], -1, 2).permute(2, 3, 0, 1, 4)
    return torch.irfft(wx_fft, 2, onesided=False)