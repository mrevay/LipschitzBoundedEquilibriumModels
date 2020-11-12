import torch
import torch.nn as nn
from torch.autograd import Function
import utils
import time
import copy

import mdeq_module.broyden as broyden
import mdeq_module.deq2d as deq

import matplotlib.pyplot as plt
import numpy as np


class Broyden(nn.Module):

    def __init__(self, linear_module, nonlin_module, alpha=1.0, tol=1e-5, max_iter=50, verbose=True):
        super().__init__()
        self.linear_module = linear_module
        self.nonlin_module = nonlin_module
        self.alpha = alpha
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose
        self.stats = utils.SplittingMethodStats()
        self.save_abs_err = False

    def forward(self, x, z0=None):
        """ Forward pass of the MON, find equilibrium using FISTA"""

        start = time.time()

        with torch.no_grad():
            xt = (x, )
            z = tuple(z0i for z0i in z0) if z0 is not None else \
                tuple(torch.zeros(s, dtype=x.dtype, device=x.device)
                      for s in self.linear_module.z_shape(x.shape[0]))

            n = len(z)

            err = 1.0
            it = 0
            errs = []

            # Calulate the roots of the model
            threshold = 50
            nelem = sum([elem.nelement() for elem in z])
            eps = 1e-7 * np.sqrt(nelem)

            def eval_net(z, u, *args):
                zlin = self.linear_module(u, *z)[0]
                return self.nonlin_module(zlin)

            z_est = deq.DEQFunc2d.broyden_find_root(
                eval_net, z, x, eps, threshold, -1, None)

            # bias = self.linear_module.bias(x)[0]

            # cutoffs = [(elem.size(1), elem.size(2), elem.size(3))
            #            for elem in z]
            # # Initial guess
            # z1_est = deq.DEQFunc2d.list2vec(z)

            # def g(z):
            #     zp = deq.DEQFunc2d.vec2list(z, cutoffs)
            #     zlin = self.linear_module(x, *zp)[0]
            #     dz = (self.nonlin_module(zlin)[0] - zp[0], )
            #     return deq.DEQFunc2d.list2vec(dz)

            # res = broyden.broyden(g, z1_est, threshold,
            #                       eps, name="forward", ls=True)
            # z = deq.DEQFunc2d.vec2list(res['result'], cutoffs)

        # Run the forward pass one more time, tracking gradients, then backward placeholder
        zn = self.linear_module(x, *z_est)
        zn = self.nonlin_module(*zn)

        if self.verbose:
            # err = z_est[0] - eval_net(z_est, x)[0]
            err = zn[0] - z_est[0]
            rel_err = err.norm().item() / (1E-6 + z_est[0].norm().item())
            # print("Forward: ", res['nstep'], rel_err)
            print("Forward: ", rel_err)

        # Run backwards pass. Currently uses forward backward splitting.
        zn = self.Backward.apply(self, *zn)
        self.stats.fwd_iters.update(it)
        self.stats.fwd_time.update(time.time() - start)
        self.errs = errs
        return zn

    class Backward(Function):
        @ staticmethod
        def forward(ctx, splitter, *z):
            ctx.splitter = splitter
            ctx.save_for_backward(*z)
            return z

        @ staticmethod
        def backward(ctx, *g):
            start = time.time()
            sp = ctx.splitter
            n = len(g)
            z = ctx.saved_tensors
            j = sp.nonlin_module.derivative(*z)
            I = [j[i] == 0 for i in range(n)]
            d = [(1 - j[i]) / j[i] for i in range(n)]
            v = tuple(j[i] * g[i] for i in range(n))
            u = tuple(torch.zeros(s, dtype=g[0].dtype, device=g[0].device)
                      for s in sp.linear_module.z_shape(g[0].shape[0]))

            err = 1.0
            it = 0
            errs = []
            while (err > sp.tol and it < sp.max_iter):
                un = sp.linear_module.multiply_transpose(*u)
                un = tuple((1 - sp.alpha) * u[i] +
                           sp.alpha * un[i] for i in range(n))
                un = tuple((un[i] + sp.alpha * (1 + d[i]) * v[i]) /
                           (1 + sp.alpha * d[i]) for i in range(n))
                for i in range(n):
                    un[i][I[i]] = v[i][I[i]]

                err = sum((un[i] - u[i]).norm().item() /
                          (1e-6 + un[i].norm().item()) for i in range(n))
                errs.append(err)
                u = un
                it = it + 1

            if sp.verbose:
                print("Backward: ", it, err)

            dg = sp.linear_module.multiply_transpose(*u)
            dg = tuple(g[i] + dg[i] for i in range(n))

            sp.stats.bkwd_iters.update(it)
            sp.stats.bkwd_time.update(time.time() - start)
            sp.errs = errs
            return (None,) + dg


class FISTA(nn.Module):

    def __init__(self, linear_module, nonlin_module, alpha=1.0, tol=1e-5, max_iter=5000, verbose=True):
        super().__init__()
        self.linear_module = linear_module
        self.nonlin_module = nonlin_module
        self.alpha = alpha
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose
        self.stats = utils.SplittingMethodStats()
        self.save_abs_err = False

        self.rho = None
        self.last_iters = None

    def forward(self, x, z0=None):
        """ Forward pass of the MON, find equilibrium using FISTA"""

        start = time.time()

        # During training we need to keep track of operator norm.
        # During eval it has been fixed.
        if self.training:
            # Estimate Lipschitz constant of (I-W) using power method.
            self.rho = self.estimate_operator_norm(epsilon=1E-4)

        with torch.no_grad():

            yk = tuple(z0i for z0i in z0) if z0 is not None else \
                tuple(torch.zeros(s, dtype=x.dtype, device=x.device)
                      for s in self.linear_module.z_shape(x.shape[0]))

            xk = tuple(torch.zeros(s, dtype=x.dtype, device=x.device)
                       for s in self.linear_module.z_shape(x.shape[0]))
            tk = torch.ones((1), device=x.device)

            n = len(yk)
            bias = self.linear_module.bias(x)

            def eval_prox(z, rho=self.rho):
                Wz = self.linear_module.multiply(*z)
                Z = tuple((1-1/rho) * z[i] + Wz[i] /
                          rho + bias[i] / rho for i in range(n))
                return self.nonlin_module(*Z)

            err = 1.0
            it = 0
            errs = []
            t0 = time.time()
            while (err > self.tol and it < self.max_iter):

                # FISTA Algorithm
                xkp = eval_prox(yk)
                tkp = 0.5 * (1 + torch.sqrt(1 + 4 * tk ** 2))

                yk = tuple(xkp[i] + (tk - 1) / (tkp) * (xkp[i] - xk[i])
                           for i in range(n))
                xk = xkp

                fn = self.nonlin_module(*self.linear_module(x, *yk))
                err = sum((yk[i] - fn[i]).norm().item() /
                          (1e-6 + yk[i].norm().item()) for i in range(n))
                errs.append(err)
                z = yk
                it = it + 1

        # Adapt rho based on number of iterations.

        if self.verbose:
            dt = time.time() - t0
            print(f"Forward: \titer={it:d}, err={err:1.4f}, time={dt:1.3f}s")

        # Run the forward pass one more time, tracking gradients, then backward placeholder
        zn = self.linear_module(x, *z)
        zn = self.nonlin_module(*zn)

        zn = self.Backward.apply(self, *zn)
        self.stats.fwd_iters.update(it)
        self.stats.fwd_time.update(time.time() - start)
        self.errs = errs
        return zn

    def estimate_operator_norm(self, epsilon, max_iters=1000):
        # Estimates the operator norm of I-W where W is the forward part of the linear module.
        # This method essentially just uses a power method.

        device = self.linear_module.U.weight.device

        module_copy = copy.deepcopy(self.linear_module)  # get a new instance

        # Sampling initial vector
        u = torch.randn(self.linear_module.z_shape(
            1)[0], requires_grad=True, device=device)

        op_norm = 0

        for ii in range(max_iters):
            u.data = u.data / u.data.norm()
            v = (u[0] - module_copy.multiply(u)[0]).view(-1, 1)
            vsq = 0.5 * (v.T @ v)

            vsq.backward(retain_graph=True)
            u.data = u.grad.clone()

            u.grad.data.zero_()

            op_norm_last = op_norm
            op_norm = u.view(-1).norm().sqrt()

            if abs(op_norm - op_norm_last) / op_norm < epsilon:
                return op_norm

        print('Operator norm estimation did not converge.')
        return op_norm.item()

    class Backward(Function):
        @ staticmethod
        def forward(ctx, splitter, *z):
            ctx.splitter = splitter
            ctx.save_for_backward(*z)
            return z

        @ staticmethod
        def backward(ctx, *g):
            start = time.time()
            sp = ctx.splitter
            n = len(g)
            z = ctx.saved_tensors
            j = sp.nonlin_module.derivative(*z)
            I = [j[i] == 0 for i in range(n)]
            d = [(1 - j[i]) / j[i] for i in range(n)]
            v = tuple(j[i] * g[i] for i in range(n))
            u = tuple(torch.zeros(s, dtype=g[0].dtype, device=g[0].device)
                      for s in sp.linear_module.z_shape(g[0].shape[0]))

            err = 1.0
            it = 0
            errs = []

            t0 = time.time()
            while (err > sp.tol and it < sp.max_iter):
                un = sp.linear_module.multiply_transpose(*u)
                un = tuple((1 - sp.alpha) * u[i] +
                           sp.alpha * un[i] for i in range(n))
                un = tuple((un[i] + sp.alpha * (1 + d[i]) * v[i]) /
                           (1 + sp.alpha * d[i]) for i in range(n))
                for i in range(n):
                    un[i][I[i]] = v[i][I[i]]

                err = sum((un[i] - u[i]).norm().item() /
                          (1e-6 + un[i].norm().item()) for i in range(n))
                errs.append(err)
                u = un
                it = it + 1

            if sp.verbose:
                # print("Backward: ", it, err)

                dt = time.time() - t0
                print(
                    f"Backward: \titer={it:d}, err={err:1.4f}, time={dt:1.3f}s")

            dg = sp.linear_module.multiply_transpose(*u)
            dg = tuple(g[i] + dg[i] for i in range(n))

            sp.stats.bkwd_iters.update(it)
            sp.stats.bkwd_time.update(time.time() - start)
            sp.errs = errs
            return (None,) + dg


class MONForwardBackwardSplitting(nn.Module):

    def __init__(self, linear_module, nonlin_module, alpha=1.0, tol=1e-5, max_iter=50, verbose=False):
        super().__init__()
        self.linear_module = linear_module
        self.nonlin_module = nonlin_module
        self.alpha = alpha
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose
        self.stats = utils.SplittingMethodStats()
        self.save_abs_err = False

    def forward(self, x):
        """ Forward pass of the MON, find an equilibirum with forward-backward splitting"""

        start = time.time()
        # Run the forward pass _without_ tracking gradients
        with torch.no_grad():
            z = tuple(torch.zeros(s, dtype=x.dtype, device=x.device)
                      for s in self.linear_module.z_shape(x.shape[0]))
            n = len(z)
            bias = self.linear_module.bias(x)

            err = 1.0
            it = 0
            errs = []
            while (err > self.tol and it < self.max_iter):
                zn = self.linear_module.multiply(*z)
                zn = tuple(
                    (1 - self.alpha) * z[i] + self.alpha * (zn[i] + bias[i]) for i in range(n))
                zn = self.nonlin_module(*zn)
                if self.save_abs_err:
                    fn = self.nonlin_module(*self.linear_module(x, *zn))
                    err = sum((zn[i] - fn[i]).norm().item() /
                              (zn[i].norm().item()) for i in range(n))
                    errs.append(err)
                else:
                    err = sum((zn[i] - z[i]).norm().item() /
                              (1e-6 + zn[i].norm().item()) for i in range(n))
                z = zn
                it = it + 1

        if self.verbose:
            print("Forward: ", it, err)

        # Run the forward pass one more time, tracking gradients, then backward placeholder
        zn = self.linear_module(x, *z)
        zn = self.nonlin_module(*zn)
        zn = self.Backward.apply(self, *zn)
        self.stats.fwd_iters.update(it)
        self.stats.fwd_time.update(time.time() - start)
        self.errs = errs
        return zn

    class Backward(Function):
        @ staticmethod
        def forward(ctx, splitter, *z):
            ctx.splitter = splitter
            ctx.save_for_backward(*z)
            return z

        @ staticmethod
        def backward(ctx, *g):
            start = time.time()
            sp = ctx.splitter
            n = len(g)
            z = ctx.saved_tensors
            j = sp.nonlin_module.derivative(*z)
            I = [j[i] == 0 for i in range(n)]
            d = [(1 - j[i]) / j[i] for i in range(n)]
            v = tuple(j[i] * g[i] for i in range(n))
            u = tuple(torch.zeros(s, dtype=g[0].dtype, device=g[0].device)
                      for s in sp.linear_module.z_shape(g[0].shape[0]))

            err = 1.0
            it = 0
            errs = []
            while (err > sp.tol and it < sp.max_iter):
                un = sp.linear_module.multiply_transpose(*u)
                un = tuple((1 - sp.alpha) * u[i] +
                           sp.alpha * un[i] for i in range(n))
                un = tuple((un[i] + sp.alpha * (1 + d[i]) * v[i]) /
                           (1 + sp.alpha * d[i]) for i in range(n))
                for i in range(n):
                    un[i][I[i]] = v[i][I[i]]

                err = sum((un[i] - u[i]).norm().item() /
                          (1e-6 + un[i].norm().item()) for i in range(n))
                errs.append(err)
                u = un
                it = it + 1

            if sp.verbose:
                print("Backward: ", it, err)

            dg = sp.linear_module.multiply_transpose(*u)
            dg = tuple(g[i] + dg[i] for i in range(n))

            sp.stats.bkwd_iters.update(it)
            sp.stats.bkwd_time.update(time.time() - start)
            sp.errs = errs
            return (None,) + dg


class MONPeacemanRachford(nn.Module):

    def __init__(self, linear_module, nonlin_module, alpha=1.0, tol=1e-5, max_iter=50, verbose=True):
        super().__init__()
        self.linear_module = linear_module
        self.nonlin_module = nonlin_module
        self.alpha = alpha
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose
        self.stats = utils.SplittingMethodStats()
        self.save_abs_err = False

    def forward(self, x):
        """ Forward pass of the MON, find an equilibirum with forward-backward splitting"""

        start = time.time()
        # Run the forward pass _without_ tracking gradients
        self.linear_module.init_inverse(1 + self.alpha, -self.alpha)
        with torch.no_grad():
            z = tuple(torch.zeros(s, dtype=x.dtype, device=x.device)
                      for s in self.linear_module.z_shape(x.shape[0]))
            u = tuple(torch.zeros(s, dtype=x.dtype, device=x.device)
                      for s in self.linear_module.z_shape(x.shape[0]))

            n = len(z)
            bias = self.linear_module.bias(x)

            err = 1.0
            it = 0
            errs = []
            while (err > self.tol and it < self.max_iter):
                u_12 = tuple(2 * z[i] - u[i] for i in range(n))
                z_12 = self.linear_module.inverse(
                    *tuple(u_12[i] + self.alpha * bias[i] for i in range(n)))
                u = tuple(2 * z_12[i] - u_12[i] for i in range(n))
                zn = self.nonlin_module(*u)

                if self.save_abs_err:
                    fn = self.nonlin_module(*self.linear_module(x, *zn))
                    err = sum((zn[i] - fn[i]).norm().item() /
                              (zn[i].norm().item()) for i in range(n))
                    errs.append(err)
                else:
                    err = sum((zn[i] - z[i]).norm().item() /
                              (1e-6 + zn[i].norm().item()) for i in range(n))
                z = zn
                it = it + 1

        if self.verbose:
            print("Forward: ", it, err)

        # Run the forward pass one more time, tracking gradients, then backward placeholder
        zn = self.linear_module(x, *z)
        zn = self.nonlin_module(*zn)

        zn = self.Backward.apply(self, *zn)
        self.stats.fwd_iters.update(it)
        self.stats.fwd_time.update(time.time() - start)
        self.errs = errs
        return zn

    class Backward(Function):
        @ staticmethod
        def forward(ctx, splitter, *z):
            ctx.splitter = splitter
            ctx.save_for_backward(*z)
            return z

        @ staticmethod
        def backward(ctx, *g):
            start = time.time()
            sp = ctx.splitter
            n = len(g)
            z = ctx.saved_tensors
            j = sp.nonlin_module.derivative(*z)
            I = [j[i] == 0 for i in range(n)]
            d = [(1 - j[i]) / j[i] for i in range(n)]
            v = tuple(j[i] * g[i] for i in range(n))

            z = tuple(torch.zeros(s, dtype=g[0].dtype, device=g[0].device)
                      for s in sp.linear_module.z_shape(g[0].shape[0]))
            u = tuple(torch.zeros(s, dtype=g[0].dtype, device=g[0].device)
                      for s in sp.linear_module.z_shape(g[0].shape[0]))

            err = 1.0
            errs = []
            it = 0
            while (err > sp.tol and it < sp.max_iter):
                u_12 = tuple(2 * z[i] - u[i] for i in range(n))
                z_12 = sp.linear_module.inverse_transpose(*u_12)
                u = tuple(2 * z_12[i] - u_12[i] for i in range(n))
                zn = tuple((u[i] + sp.alpha * (1 + d[i]) * v[i]) /
                           (1 + sp.alpha * d[i]) for i in range(n))
                for i in range(n):
                    zn[i][I[i]] = v[i][I[i]]

                err = sum((zn[i] - z[i]).norm().item() /
                          (1e-6 + zn[i].norm().item()) for i in range(n))
                errs.append(err)
                z = zn
                it = it + 1

            if sp.verbose:
                print("Backward: ", it, err)

            dg = sp.linear_module.multiply_transpose(*zn)
            dg = tuple(g[i] + dg[i] for i in range(n))

            sp.stats.bkwd_iters.update(it)
            sp.stats.bkwd_time.update(time.time() - start)
            sp.errs = errs
            return (None,) + dg


if __name__ == "__main__":

    # Test Operator norm estimation
    n = 50
    W = torch.randn(n, n)

    # Calculate spectral norm directly.
    sn = W.svd()[1].max()
    print("sn calculated via SVD is: ", sn)

    # Sampling method
    u = torch.nn.Parameter(torch.randn(n, 1))

    for ii in range(100):
        u.data = u.data / u.data.norm()
        v = W @ u
        vsq = 0.5 * (v.T @ v)

        vsq.backward(retain_graph=True)
        u.data = u.grad.clone()

        u.grad.data.zero_()

        print("iterative sn is: ", u.norm().sqrt())
