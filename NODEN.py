import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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

        # BTB = self.B.weight.T @ self.Lambda.diag() @ self.Lambda.diag() @ self.B.weight

        # LambdaInv = 1 / self.Lambda
        # W = Id - LambdaInv.diag() @ (S.T - S - GTG / self.gamma
        #                              - BTB / self.gamma - VTV)
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
