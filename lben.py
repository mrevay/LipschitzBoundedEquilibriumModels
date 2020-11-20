import torch
import torch.nn as nn
import torch.nn.functional as F


class LBEN_FC(nn.Module):
    """ Simple MON linear class, just a single full multiply. """

    def __init__(self, in_dim, out_dim, m=1.0):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.U = nn.Linear(in_dim, out_dim)
        self.V = nn.Linear(out_dim, out_dim, bias=False)
        self.S = nn.Linear(out_dim, out_dim, bias=False)

        self.psi = nn.Parameter(torch.zeros((out_dim)))
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


class LBEN_Lip_FC(nn.Module):
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

        Psi_inv = (1 / Psi).diag()
        BBT = Psi_inv @ self.B.weight @ self.B.weight.T @ Psi_inv

        W = Id - Psi.diag() @ (GTG / 2 / self.gamma + BBT / 2 / self.gamma + VTV + S.T - S)
        return W


class LBEN_Conv(nn.Module):
    """ Simple MON linear class, just a single full multiply. """

    def __init__(self, in_dim, in_channels, out_channels, shp, kernel_size=3, m=1.0, metric="full"):
        super().__init__()
        self.U = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.A = nn.Conv2d(out_channels, out_channels, kernel_size, bias=False)
        self.g = nn.Parameter(torch.tensor(1.))
        self.h = nn.Parameter(torch.tensor(1.))
        self.B = nn.Conv2d(out_channels, out_channels, kernel_size, bias=False)
        self.pad = 4 * ((kernel_size - 1) // 2,)
        self.shp = shp
        self.m = m

        self.metric = metric
        if metric == "full":
            self.psi = nn.Parameter(torch.zeros(
                (1, out_channels, shp[0], shp[1])))

        elif metric == "channels":
            self.psi = nn.Parameter(torch.zeros(
                (1, out_channels, 1, 1)))

        elif metric == "image":
            self.psi = nn.Parameter(torch.zeros(
                (1, 1, shp[0], shp[1])))

        elif metric == "identity":
            self.psi = nn.Parameter(torch.zeros(
                (1, 1, 1, 1)))

        else:
            raise Exception("Invalid metric chosen")

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

        if self.metric == "identity":
            Psi = torch.ones_like(self.psi)
        else:
            Psi = torch.exp(self.psi)

        A = self.A.weight / self.A.weight.view(-1).norm()
        B = self.h * self.B.weight / self.B.weight.view(-1).norm()

        Az = F.conv2d(self.cpad(z[0]), A)
        ATAz = self.uncpad(F.conv_transpose2d(self.cpad(Az), A))
        Bz = F.conv2d(self.cpad(z[0]), B)
        BTz = self.uncpad(F.conv_transpose2d(self.cpad(z[0]), B))
        # z_out = z[0] - Psi*(self.m*z[0] + self.g*ATAz - Bz + BTz)
        z_out = z[0] - self.m * Psi * z[0] - Psi*(self.g*ATAz - Bz + BTz)
        return (z_out,)

    def multiply_transpose(self, *g):

        if self.metric == "identity":
            Psi = torch.ones_like(self.psi)
        else:
            Psi = torch.exp(self.psi)

        PsiG = Psi * g[0]
        A = self.A.weight / self.A.weight.view(-1).norm()
        B = self.h * self.B.weight / self.B.weight.view(-1).norm()

        Ag = F.conv2d(self.cpad(PsiG), A)
        ATAg = self.uncpad(F.conv_transpose2d(self.cpad(Ag), A))
        Bg = F.conv2d(self.cpad(PsiG), B)
        BTg = self.uncpad(F.conv_transpose2d(self.cpad(PsiG), B))
        # g_out = g[0] - (self.m*PsiG[0] + self.g * ATAg + Bg - BTg)
        g_out = g[0] - self.m*PsiG - (self.g * ATAg + Bg - BTg)
        return (g_out,)


class LBEN_Conv_Test_Init(nn.Module):
    """ Simple MON linear class, just a single full multiply. """

    def __init__(self, in_dim, in_channels, out_channels, shp, kernel_size=3, m=1.0):
        super().__init__()
        self.U = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.A = nn.Conv2d(out_channels, out_channels, kernel_size, bias=False)
        self.A.weight.data /= self.A.weight.view(-1).norm()
        self.B = nn.Conv2d(out_channels, out_channels, kernel_size, bias=False)
        self.B.weight.data /= self.B.weight.view(-1).norm()

        self.pad = 4 * ((kernel_size - 1) // 2,)
        self.shp = shp
        self.m = m

        self.psi = nn.Parameter(torch.zeros(
            (1, out_channels, shp[0], shp[1])))

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
        A = self.A.weight
        B = self.B.weight

        Az = F.conv2d(self.cpad(z[0]), A)
        ATAz = self.uncpad(F.conv_transpose2d(self.cpad(Az), A))
        Bz = F.conv2d(self.cpad(z[0]), B)
        BTz = self.uncpad(F.conv_transpose2d(self.cpad(z[0]), B))
        # z_out = z[0] - Psi*(self.m*z[0] + self.g*ATAz - Bz + BTz)
        z_out = z[0] - self.m * Psi * z[0] - Psi*(ATAz - Bz + BTz)
        return (z_out,)

    def multiply_transpose(self, *g):
        Psi = torch.exp(self.psi)
        PsiG = Psi * g[0]
        A = self.A.weight
        B = self.B.weight

        Ag = F.conv2d(self.cpad(PsiG), A)
        ATAg = self.uncpad(F.conv_transpose2d(self.cpad(Ag), A))
        Bg = F.conv2d(self.cpad(PsiG), B)
        BTg = self.uncpad(F.conv_transpose2d(self.cpad(PsiG), B))
        # g_out = g[0] - (self.m*PsiG[0] + self.g * ATAg + Bg - BTg)
        g_out = g[0] - self.m * PsiG - (ATAg + Bg - BTg)
        return (g_out,)


class LBEN_Lip_Conv(nn.Module):
    """ MON class with a single 3x3 (circular) convolution """

    def __init__(self, in_channels, channels, out_dim, gamma, shp, kernel_size=3, m=1.0, pool=4):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = channels

        self.U = nn.Conv2d(in_channels, channels, kernel_size)
        self.A = nn.Conv2d(channels, channels, kernel_size, bias=False)
        self.g = nn.Parameter(torch.tensor(1.))
        self.h = nn.Parameter(torch.tensor(1.))

        self.B = nn.Conv2d(channels, channels, kernel_size, bias=False)
        self.pad = 4 * ((kernel_size - 1) // 2,)
        self.shp = shp
        self.m = m

        self.psi = nn.Parameter(torch.zeros((channels, shp[0], shp[1])))

        self.gamma = gamma
        self.pool = pool

        n = shp[0]
        self.out_dim = channels * (n // pool) ** 2
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

        Psi = torch.exp(self.psi)[None, ...]
        # A = self.A.weight
        # B = self.B.weight
        A = self.A.weight / self.A.weight.view(-1).norm()
        B = self.h * self.B.weight / self.B.weight.view(-1).norm()

        U = self.U.weight

        Az = F.conv2d(self.cpad(z[0]), A)
        ATAz = self.uncpad(F.conv_transpose2d(self.cpad(Az), A))
        Bz = F.conv2d(self.cpad(z[0]), B)
        BTz = self.uncpad(F.conv_transpose2d(self.cpad(z[0]), B))

        #  Calculate U U' psi^{-1} z
        UTz = F.conv_transpose2d(z[0]/Psi, U)
        UUTz = F.conv2d(UTz, U)

        G = self.Wout.weight
        zp = F.avg_pool2d(z[-1], self.pool)
        zpvec = zp.view(z[0].shape[0], -1)

        pTGTGpz = self.pool_adjoint(((zpvec @ G.T) @ G).view_as(zp))

        z_out = z[0] - 0.5 * UUTz / self.gamma  \
            - Psi * (self.m * z[0] + self.g*ATAz - Bz + BTz +
                     0.5 * pTGTGpz.view_as(z[0]) / self.gamma)

        return (z_out,)

    def multiply_transpose(self, *g):
        # A = self.A.weight / self.A.weight.view(-1).norm()
        # A = self.A.weight
        # B = self.B.weight
        A = self.A.weight / self.A.weight.view(-1).norm()
        B = self.h * self.B.weight / self.B.weight.view(-1).norm()

        U = self.U.weight

        Psi = torch.exp(self.psi)[None, ...]
        # Psi = torch.ones_like(self.psi)[None, ...]
        gpsi = g[0] * Psi

        Ag = F.conv2d(self.cpad(gpsi), A)
        ATAg = self.uncpad(F.conv_transpose2d(self.cpad(Ag), A))
        Bg = F.conv2d(self.cpad(gpsi), B)
        BTg = self.uncpad(F.conv_transpose2d(self.cpad(gpsi), B))

        # Calculate Psi^{-1} U U' z
        UTz = F.conv_transpose2d(g[0], U)
        # UUTz = self.uncpad(F.conv2d(self.cpad(UTz), U)) / Psi
        UUTz = F.conv2d(UTz, U) / Psi

        G = self.Wout.weight

        gp = F.avg_pool2d(gpsi, self.pool)
        gpvec = gp.view(gpsi.shape[0], -1)

        pTGTGpz = self.pool_adjoint(((gpvec @ G.T) @ G).view_as(gp))

        g_out = g[0] - self.m * gpsi - self.g * ATAg - Bg + BTg - \
            0.5 * UUTz / self.gamma - \
            0.5 * pTGTGpz.view_as(g[0]) / self.gamma

        return (g_out,)


class LBEN_Lip_Conv_V2(nn.Module):
    """ MON class with a single 3x3 (circular) convolution """

    def __init__(self, in_channels, channels, out_dim, gamma, shp, metric="full", kernel_size=3, m=1.0, pool=4):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = channels

        self.U = nn.Conv2d(in_channels, channels, kernel_size)
        self.A = nn.Conv2d(channels, channels, kernel_size, bias=False)
        self.B = nn.Conv2d(channels, channels, kernel_size, bias=False)

        self.a = nn.Parameter(torch.tensor(1.))
        self.b = nn.Parameter(torch.tensor(1.))
        self.g = nn.Parameter(torch.tensor(1.))
        self.u = nn.Parameter(torch.tensor(1.))

        self.gamma = gamma
        self.pool = pool

        n = shp[0]
        self.out_dim = channels * (n // pool) ** 2
        self.Wout = nn.Linear(self.out_dim, 10)

        self.pad = 4 * ((kernel_size - 1) // 2,)
        self.shp = shp
        self.m = m

        self.metric = metric
        if metric == "full":
            self.psi = nn.Parameter(torch.zeros(
                (1, self.out_channels, shp[0], shp[1])))

        elif metric == "channels":
            self.psi = nn.Parameter(torch.zeros(
                (1, self.out_channels, 1, 1)))

        elif metric == "image":
            self.psi = nn.Parameter(torch.zeros(
                (1, 1, shp[0], shp[1])))

        elif metric == "identity":
            self.psi = nn.Parameter(torch.zeros(
                (1, 1, 1, 1)))

        else:
            raise Exception("Invalid metric chosen")

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
        U = torch.sqrt(self.u) * self.U.weight / self.U.weight.view(-1).norm()
        return (F.conv2d(self.cpad(x), U, self.U.bias) + self.multiply(*z)[0],)

    def bias(self, x):
        U = torch.sqrt(self.u) * self.U.weight / self.U.weight.view(-1).norm()
        return (F.conv2d(self.cpad(x), U, self.U.bias),)

    def pool_adjoint(self, x):
        'Calculate the adjoint of average pooling operator'

        assert self.shp[0] % self.pool == 0, "Pooling factor must be multiple of padded image width"

        adj = torch.nn.functional.interpolate(
            x, size=self.shp) / self.pool**2
        return adj

    def multiply(self, *z):

        if self.metric == "identity":
            Psi = torch.ones_like(self.psi)
        else:
            Psi = torch.exp(self.psi)

        # Psi = torch.exp(self.psi)[None, ...]
        # A = self.A.weight
        # B = self.B.weight
        A = self.A.weight / self.A.weight.view(-1).norm()
        B = self.b * self.B.weight / self.B.weight.view(-1).norm()
        U = self.U.weight / self.U.weight.view(-1).norm()

        Az = F.conv2d(self.cpad(z[0]), A)
        ATAz = self.uncpad(F.conv_transpose2d(self.cpad(Az), A))
        Bz = F.conv2d(self.cpad(z[0]), B)
        BTz = self.uncpad(F.conv_transpose2d(self.cpad(z[0]), B))

        #  Calculate U U' psi^{-1} z
        UTz = F.conv_transpose2d(z[0]/Psi, U)
        UUTz = F.conv2d(UTz, U)

        G = self.Wout.weight / self.Wout.weight.view(-1).norm()
        zp = F.avg_pool2d(z[-1], self.pool)
        zpvec = zp.view(z[0].shape[0], -1)

        pTGTGpz = self.pool_adjoint(
            ((zpvec @ G.T) @ G).view_as(zp)).view_as(z[0])

        z_out = z[0] - 0.5 * self.u * UUTz / self.gamma  \
            - Psi * (self.m * z[0] + self.a*ATAz - Bz + BTz +
                     0.5 * self.g * pTGTGpz / self.gamma)

        return (z_out,)

    def multiply_transpose(self, *g):
        # A = self.A.weight / self.A.weight.view(-1).norm()
        # A = self.A.weight
        # B = self.B.weight
        A = self.A.weight / self.A.weight.view(-1).norm()
        B = self.b * self.B.weight / self.B.weight.view(-1).norm()
        U = self.U.weight / self.U.weight.view(-1).norm()

        if self.metric == "identity":
            Psi = torch.ones_like(self.psi)
        else:
            Psi = torch.exp(self.psi)

        gpsi = g[0] * Psi

        Ag = F.conv2d(self.cpad(gpsi), A)
        ATAg = self.uncpad(F.conv_transpose2d(self.cpad(Ag), A))
        Bg = F.conv2d(self.cpad(gpsi), B)
        BTg = self.uncpad(F.conv_transpose2d(self.cpad(gpsi), B))

        # Calculate Psi^{-1} U U' z
        UTz = F.conv_transpose2d(g[0], U)
        # UUTz = self.uncpad(F.conv2d(self.cpad(UTz), U)) / Psi
        UUTz = F.conv2d(UTz, U) / Psi

        G = self.Wout.weight / self.Wout.weight.view(-1).norm()

        gp = F.avg_pool2d(gpsi, self.pool)
        gpvec = gp.view(gpsi.shape[0], -1)

        pTGTGpz = self.pool_adjoint(
            ((gpvec @ G.T) @ G).view_as(gp)).view_as(g[0])

        g_out = g[0] - self.m * gpsi - self.a * ATAg - Bg + BTg - \
            0.5 * self.u * UUTz / self.gamma - \
            0.5 * self.g * pTGTGpz / self.gamma

        return (g_out,)


if __name__ == "__main__":

    torch.set_default_tensor_type(torch.DoubleTensor)

    def inner(x, y):
        xp = x.reshape(x.shape[0], -1)
        yp = y.reshape(y.shape[0], -1)
        return (xp * yp).sum(dim=1)

    # Check multiply and multiply transpose for fully connected first
    n = 5
    p = 20
    batches = 100
    input_channels = 3

    z = torch.randn(batches, p)

    # Fully connected
    model = LBEN_FC(n, p)
    Wz = model.multiply(z)
    WTWz = model.multiply_transpose(Wz[0])

    print("Error in Adjoint for FC is: ", sum(
        inner(z, WTWz[0]) - inner(Wz[0], Wz[0])).item())

    # Lipschitz Fully Connected
    model = LBEN_Lip_FC(n, p, 10, 1.0, 1.0)
    Wz = model.multiply(z)
    WTWz = model.multiply_transpose(Wz[0])

    print("Error in Adjoint for LIP_FC is: ", sum(
        inner(z, WTWz[0]) - inner(Wz[0], Wz[0])).item())

    # Convolutional
    z = torch.randn(batches, n, p, p)
    model = LBEN_Conv(n, input_channels, n, (p, p), m=1.0)
    model.psi.data = torch.randn_like(model.psi)
    Wz = model.multiply(z)
    WTWz = model.multiply_transpose(Wz[0])

    print("Error in Adjoint for Conv is: ", sum(
        inner(z, WTWz[0]) - inner(Wz[0], Wz[0])).item())

    # Lipschitz Bounded Convolutional
    z = torch.randn(batches, n, p, p)
    model = LBEN_Lip_Conv(input_channels, n, 10, 1E-1, (p, p), pool=4, m=1.0)
    model.psi.data = torch.randn_like(model.psi)
    Wz = model.multiply(z)
    WTWz = model.multiply_transpose(Wz[0])

    print("Error in Adjoint for Lipschitz Conv is: ", sum(
        inner(z, WTWz[0]) - inner(Wz[0], Wz[0])).item())

    # Lipschitz Bounded Convolutional
    z = torch.randn(batches, n, p, p)
    model = LBEN_Lip_Conv_V2(input_channels, n, 10,
                             1E-1, (p, p), pool=4, m=1.0)

    model.psi.data = torch.randn_like(model.psi)
    model.a.data = torch.randn(1)
    model.b.data = torch.randn(1)
    model.g.data = torch.randn(1)
    model.u.data = torch.randn(1)
    Wz = model.multiply(z)
    WTWz = model.multiply_transpose(Wz[0])

    print("Error in Adjoint for Lipschitz Conv V2 is: ", sum(
        inner(z, WTWz[0]) - inner(Wz[0], Wz[0])).item())

    # Test Convolution
    z = torch.randn(batches, n, p, p)
    model = LBEN_Conv_Test_Init(n, input_channels, n, (p, p), m=1.0)
    model.psi.data = torch.randn_like(model.psi)
    Wz = model.multiply(z)
    WTWz = model.multiply_transpose(Wz[0])

    print("Error in Adjoint for Conv test is: ", sum(
        inner(z, WTWz[0]) - inner(Wz[0], Wz[0])).item())
