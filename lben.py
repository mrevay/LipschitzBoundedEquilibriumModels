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
        z_out = z[0] - Psi*(self.m*z[0] + ATAz - Bz + BTz)
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
        g_out = g[0] - (self.m*PsiG[0] + ATAg + Bg - BTg)
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
        A = self.A.weight
        # B = self.h * self.B.weight / self.B.weight.view(-1).norm()
        B = self.B.weight
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
            - Psi * (self.m * z[0] + ATAz - Bz + BTz +
                     0.5 * pTGTGpz.view_as(z[0]) / self.gamma)

        return (z_out,)

    def multiply_transpose(self, *g):
        # A = self.A.weight / self.A.weight.view(-1).norm()
        A = self.A.weight
        B = self.B.weight
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

        g_out = g[0] - self.m * gpsi - ATAg - Bg + BTg - \
            0.5 * UUTz / self.gamma - \
            0.5 * pTGTGpz.view_as(g[0]) / self.gamma

        return (g_out,)


class LBEN_Multi_Conv(nn.Module):
    def __init__(self, in_channels, conv_channels, image_size, kernel_size=3, m=1.0):
        super().__init__()
        self.pad = 4 * ((kernel_size - 1) // 2,)
        self.conv_shp = tuple((image_size - 2 * self.pad[0]) // 2 ** i + 2 * self.pad[0]
                              for i in range(len(conv_channels)))
        self.m = m

        # create convolutional layers
        self.U = nn.Conv2d(in_channels, conv_channels[0], kernel_size)
        self.A0 = nn.ModuleList(
            [nn.Conv2d(c, c, kernel_size, bias=False) for c in conv_channels])
        self.B0 = nn.ModuleList(
            [nn.Conv2d(c, c, kernel_size, bias=False) for c in conv_channels])
        self.A_n0 = nn.ModuleList([nn.Conv2d(c1, c2, kernel_size, bias=False, stride=2)
                                   for c1, c2 in zip(conv_channels[:-1], conv_channels[1:])])

        self.g = nn.ParameterList(
            [nn.Parameter(torch.tensor(1.)) for _ in range(len(conv_channels))])
        self.gn = nn.ParameterList(
            [nn.Parameter(torch.tensor(1.)) for _ in range(len(conv_channels) - 1)])
        self.h = nn.ParameterList(
            [nn.Parameter(torch.tensor(1.)) for _ in range(len(conv_channels))])

        self.S_idx = list()
        self.S_idxT = list()
        for n in self.conv_shp:
            p = n // 2
            q = n
            idxT = list()
            _idx = [[j + (i - 1) * p for i in range(1, q + 1)]
                    for j in range(1, p + 1)]
            for i in _idx:
                for j in i:
                    idxT.append(j - 1)
            _idx = [[j + (i - 1) * p + p * q for i in range(1, q + 1)]
                    for j in range(1, p + 1)]
            for i in _idx:
                for j in i:
                    idxT.append(j - 1)
            idx = list()
            _idx = [[j + (i - 1) * q for i in range(1, p + 1)]
                    for j in range(1, q + 1)]
            for i in _idx:
                for j in i:
                    idx.append(j - 1)
            _idx = [[j + (i - 1) * q + p * q for i in range(1, p + 1)]
                    for j in range(1, q + 1)]
            for i in _idx:
                for j in i:
                    idx.append(j - 1)
            self.S_idx.append(idx)
            self.S_idxT.append(idxT)

    def A(self, i):
        return torch.sqrt(self.g[i]) * self.A0[i].weight / self.A0[i].weight.view(-1).norm()

    def A_n(self, i):
        return torch.sqrt(self.gn[i]) * self.A_n0[i].weight / self.A_n0[i].weight.view(-1).norm()

    def B(self, i):
        return self.h[i] * self.B0[i].weight / self.B0[i].weight.view(-1).norm()

    def cpad(self, x):
        return F.pad(x, self.pad, mode="circular")

    def uncpad(self, x):
        return x[:, :, 2 * self.pad[0]:-2 * self.pad[1], 2 * self.pad[2]:-2 * self.pad[3]]

    def zpad(self, x):
        return F.pad(x, (0, 1, 0, 1))

    def unzpad(self, x):
        return x[:, :, :-1, :-1]

    def unstride(self, x):
        x[:, :, :, -1] += x[:, :, :, 0]
        x[:, :, -1, :] += x[:, :, 0, :]
        return x[:, :, 1:, 1:]

    def x_shape(self, n_batch):
        return (n_batch, self.U.in_channels, self.conv_shp[0], self.conv_shp[0])

    def z_shape(self, n_batch):
        return tuple((n_batch, self.A0[i].in_channels, self.conv_shp[i], self.conv_shp[i])
                     for i in range(len(self.A0)))

    def forward(self, x, *z):
        z_out = self.multiply(*z)
        bias = self.bias(x)
        return tuple([z_out[i] + bias[i] for i in range(len(self.A0))])

    def bias(self, x):
        z_shape = self.z_shape(x.shape[0])
        n = len(self.A0)

        b_out = [self.U(self.cpad(x))]
        for i in range(n - 1):
            b_out.append(torch.zeros(z_shape[i + 1], dtype=self.A0[0].weight.dtype,
                                     device=self.A0[0].weight.device))
        return tuple(b_out)

    def multiply(self, *z):

        def multiply_zi(z1, A1, B1, A1_n=None, z0=None, A2_n=None):
            Az1 = F.conv2d(self.cpad(z1), A1)
            A1TA1z1 = self.uncpad(F.conv_transpose2d(self.cpad(Az1), A1))
            B1z1 = F.conv2d(self.cpad(z1), B1)
            B1Tz1 = self.uncpad(F.conv_transpose2d(self.cpad(z1), B1))
            out = (1 - self.m) * z1 - A1TA1z1 + B1z1 - B1Tz1
            if A2_n is not None:
                A2_nz1 = F.conv2d(self.cpad(z1), A2_n, stride=2)
                A2_nTA2_nz1 = self.unstride(F.conv_transpose2d(A2_nz1,
                                                               A2_n, stride=2))
                out -= A2_nTA2_nz1
            if A1_n is not None:
                A1_nz0 = self.zpad(F.conv2d(self.cpad(z0), A1_n, stride=2))
                A1TA1_nz0 = self.uncpad(
                    F.conv_transpose2d(self.cpad(A1_nz0), A1))
                out -= 2 * A1TA1_nz0
            return out

        n = len(self.A0)
        z_out = [multiply_zi(z[0], self.A(0), self.B(0), A2_n=self.A_n(0))]
        for i in range(1, n - 1):
            z_out.append(multiply_zi(z[i], self.A(i), self.B(i),
                                     A1_n=self.A_n(i - 1), z0=z[i - 1], A2_n=self.A_n(i)))
        z_out.append(multiply_zi(z[n - 1], self.A(n - 1), self.B(n - 1),
                                 A1_n=self.A_n(n - 2), z0=z[n - 2]))

        return tuple(z_out)

    def multiply_transpose(self, *g):

        def multiply_zi(z1, A1, B1, z2=None, A2_n=None, A2=None):
            Az1 = F.conv2d(self.cpad(z1), A1)
            A1TA1z1 = self.uncpad(F.conv_transpose2d(self.cpad(Az1), A1))
            B1z1 = F.conv2d(self.cpad(z1), B1)
            B1Tz1 = self.uncpad(F.conv_transpose2d(self.cpad(z1), B1))
            out = (1 - self.m) * z1 - A1TA1z1 - B1z1 + B1Tz1
            if A2_n is not None:
                A2z2 = F.conv2d(self.cpad(z2), A2)
                A2_nTA2z2 = self.unstride(F.conv_transpose2d(self.unzpad(A2z2),
                                                             A2_n, stride=2))

                out -= 2 * A2_nTA2z2

                A2_nz1 = F.conv2d(self.cpad(z1), A2_n, stride=2)
                A2_nTA2_nz1 = self.unstride(F.conv_transpose2d(A2_nz1,
                                                               A2_n, stride=2))

                out -= A2_nTA2_nz1

            return out

        n = len(self.A0)
        g_out = []
        for i in range(n - 1):
            g_out.append(multiply_zi(g[i], self.A(i), self.B(
                i), z2=g[i + 1], A2_n=self.A_n(i), A2=self.A(i + 1)))
        g_out.append(multiply_zi(g[n - 1], self.A(n - 1), self.B(n - 1)))

        return g_out

    def apply_inverse_conv(self, z, i):
        z0_fft = fft_to_complex_vector(torch.rfft(z, 2, onesided=False))
        y0 = 0.5 * \
            z0_fft.bmm((self.D2[i] @ self.D1inv[i]
                        ).transpose(1, 2))[self.S_idx[i]]
        n = self.conv_shp[i]
        y1 = y0[:n ** 2 // 4] + y0[n ** 2 // 4:n ** 2 // 2] + \
            y0[n ** 2 // 2:3 * n ** 2 // 4] + y0[3 * n ** 2 // 4:]
        y2 = y1.bmm(self.H[i].transpose(1, 2))
        y3 = y2.repeat(4, 1, 1)
        y4 = y3[self.S_idxT[i]]
        y5 = 0.5 * y4.bmm(self.D2[i] @ self.D1inv[i].transpose(1, 2))
        x0 = z0_fft.bmm(self.D1inv[i].transpose(1, 2)) - y5
        x0 = x0.view(n, n, x0.shape[1], -1, 2).permute(2, 3, 0, 1, 4)
        x0 = torch.irfft(x0, 2, onesided=False)
        return x0

    def apply_inverse_conv_transpose(self, g, i):
        g0_fft = fft_to_complex_vector(torch.rfft(g, 2, onesided=False))
        y0 = 0.5 * \
            g0_fft.bmm(self.D1inv[i] @
                       self.D2[i].transpose(1, 2))[self.S_idx[i]]
        n = self.conv_shp[i]
        y1 = y0[:n ** 2 // 4] + y0[n ** 2 // 4:n ** 2 // 2] + \
            y0[n ** 2 // 2:3 * n ** 2 // 4] + y0[3 * n ** 2 // 4:]
        y2 = y1.bmm(self.H[i])
        y3 = y2.repeat(4, 1, 1)
        y4 = y3[self.S_idxT[i]]
        y5 = 0.5 * y4.bmm(self.D2[i] @ self.D1inv[i])
        x0 = g0_fft.bmm(self.D1inv[i]) - y5
        x0 = x0.view(n, n, x0.shape[1], -1, 2).permute(2, 3, 0, 1, 4)
        x0 = torch.irfft(x0, 2, onesided=False)
        return x0


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
    model = LBEN_Conv(n, input_channels, n, (p, p), m=0.0)
    model.psi.data = torch.randn_like(model.psi)
    Wz = model.multiply(z)
    WTWz = model.multiply_transpose(Wz[0])

    print("Error in Adjoint for Conv is: ", sum(
        inner(z, WTWz[0]) - inner(Wz[0], Wz[0])).item())

    # Lipschitz Bounded Convolutional
    z = torch.randn(batches, n, p, p)
    model = LBEN_Lip_Conv(input_channels, n, 10, 1E-1, (p, p), pool=4)
    model.psi.data = torch.randn_like(model.psi)
    Wz = model.multiply(z)
    WTWz = model.multiply_transpose(Wz[0])

    print("Error in Adjoint for Lipschitz Conv is: ", sum(
        inner(z, WTWz[0]) - inner(Wz[0], Wz[0])).item())
