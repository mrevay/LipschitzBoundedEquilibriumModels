import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import mon
import numpy as np
import time

import foolbox as fb

import matplotlib.pyplot as plt

import utils
import NODEN

# torch.set_default_dtype(torch.float64)  # double precision for SDP
# torch.set_printoptions(precision=4)


def cuda(tensor):
    if torch.cuda.is_available():
        return tensor.cuda()
    else:
        return tensor


def train(trainLoader, testLoader, model, epochs=15, max_lr=1e-3,
          print_freq=10, change_mo=True, model_path=None, lr_mode='step',
          step=10, tune_alpha=False, max_alpha=1.):

    optimizer = optim.Adam(model.parameters(), lr=max_lr)

    if lr_mode == '1cycle':
        def lr_schedule(t): return np.interp([t],
                                             [0, (epochs-5)//2,
                                              epochs-5, epochs],
                                             [1e-3, max_lr, 1e-3, 1e-3])[0]
    elif lr_mode == 'step':
        lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer, step, gamma=0.1, last_epoch=-1)
    elif lr_mode != 'constant':
        raise Exception('lr mode one of constant, step, 1cycle')

    if change_mo:
        max_mo = 0.85

        def momentum_schedule(t): return np.interp([t],
                                                   [0, (epochs - 5) // 2,
                                                    epochs - 5, epochs],
                                                   [0.95, max_mo, 0.95, 0.95])[0]

    train_loss = []
    val_loss = []

    model = cuda(model)

    for epoch in range(1, 1 + epochs):
        tloss = 0

        nProcessed = 0
        incorrect_train = 0
        nTrain = len(trainLoader.dataset)
        model.train()
        start = time.time()
        for batch_idx, batch in enumerate(trainLoader):
            if (batch_idx == 30 or batch_idx == int(len(trainLoader)/2)) and tune_alpha:
                run_tune_alpha(model, cuda(batch[0]), max_alpha)
            if lr_mode == '1cycle':
                lr = lr_schedule(epoch - 1 + batch_idx / len(trainLoader))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            if change_mo:
                beta1 = momentum_schedule(
                    epoch - 1 + batch_idx / len(trainLoader))
                for param_group in optimizer.param_groups:
                    param_group['betas'] = (
                        beta1, optimizer.param_groups[0]['betas'][1])

            data, target = cuda(batch[0]), cuda(batch[1])
            optimizer.zero_grad()
            preds = model(data)
            ce_loss = nn.CrossEntropyLoss()(preds, target)
            tloss += ce_loss.item()

            ce_loss.backward()
            nProcessed += len(data)
            incorrect_train += preds.float().argmax(1).ne(target.data).sum()

            if batch_idx % print_freq == 0 and batch_idx > 0:
                incorrect = preds.float().argmax(1).ne(target.data).sum()
                err = 100. * incorrect.float() / float(len(data))
                partialEpoch = epoch + batch_idx / len(trainLoader) - 1
                print('Train Epoch: {:.2f} [{}/{} ({:.0f}%)]\tLoss: {:.4f}\tError: {:.2f}'.format(
                    partialEpoch, nProcessed, nTrain,
                    100. * batch_idx / len(trainLoader),
                    ce_loss.item(), err))
                model.mon.stats.report()
                model.mon.stats.reset()

            optimizer.step()

            if hasattr(model.mon.linear_module, 'Lambda'):
                model.mon.linear_module.Lambda.data[model.mon.linear_module.Lambda.data <= 1E-3] = 1E-3
            if hasattr(model.mon.linear_module, 'Psi'):
                model.mon.linear_module.Psi.data[model.mon.linear_module.Psi.data <= 1E-3] = 1E-3

        if lr_mode == 'step':
            lr_scheduler.step()

        if model_path is not None:
            torch.save(model.state_dict(), model_path)

        # estimate_Lip(model, 50, data.shape[1], data.shape[2])
        print("Tot train time: {}".format(time.time() - start))
        train_loss.append(100. * incorrect_train.cpu().item() /
                          float(len(trainLoader.dataset)))

        start = time.time()
        v_loss = 0
        test_loss = 0
        incorrect_val = 0
        model.eval()
        with torch.no_grad():
            for batch in testLoader:
                data, target = cuda(batch[0]), cuda(batch[1])
                preds = model(data)
                ce_loss = nn.CrossEntropyLoss(reduction='sum')(preds, target)
                test_loss += ce_loss
                v_loss += ce_loss.item()
                incorrect_val += preds.float().argmax(1).ne(target.data).sum()
            test_loss /= len(testLoader.dataset)
            nTotal = len(testLoader.dataset)
            err = 100. * incorrect_val.float() / float(nTotal)
            print('\n\nTest set: Average loss: {:.4f}, Error: {}/{} ({:.2f}%)'.format(
                test_loss, incorrect_val, nTotal, err))

        val_loss.append(100. * incorrect_val.cpu().item() / float(nTotal))

        print("Tot test time: {}\n\n\n\n".format(time.time() - start))

    return train_loss, val_loss


def run_tune_alpha(model, x, max_alpha):
    print("----tuning alpha----")
    print("current: ", model.mon.alpha)
    orig_alpha = model.mon.alpha
    model.mon.stats.reset()
    model.mon.alpha = max_alpha
    with torch.no_grad():
        model(x)
    iters = model.mon.stats.fwd_iters.val
    model.mon.stats.reset()
    iters_n = iters
    print('alpha: {}\t iters: {}'.format(model.mon.alpha, iters_n))
    while model.mon.alpha > 1e-4 and iters_n <= iters:
        model.mon.alpha = model.mon.alpha/2
        with torch.no_grad():
            model(x)
        iters = iters_n
        iters_n = model.mon.stats.fwd_iters.val
        print('alpha: {}\t iters: {}'.format(model.mon.alpha, iters_n))
        model.mon.stats.reset()

    if iters == model.mon.max_iter:
        print("none converged, resetting to current")
        model.mon.alpha = orig_alpha
    else:
        model.mon.alpha = model.mon.alpha * 2
        print("setting to: ", model.mon.alpha)
    print("--------------\n")


def mnist_loaders(train_batch_size, test_batch_size=None, swap_labels=0):
    if test_batch_size is None:
        test_batch_size = train_batch_size

    trainLoader = torch.utils.data.DataLoader(
        dset.MNIST('data',
                   train=True,
                   download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
        batch_size=train_batch_size,
        shuffle=True)

    testLoader = torch.utils.data.DataLoader(
        dset.MNIST('data',
                   train=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
        batch_size=test_batch_size,
        shuffle=False)
    return trainLoader, testLoader


def cifar_loaders(train_batch_size, test_batch_size=None, augment=True):
    if test_batch_size is None:
        test_batch_size = train_batch_size

    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2470, 0.2435, 0.2616])

    if augment:
        transforms_list = [transforms.RandomHorizontalFlip(),
                           transforms.RandomCrop(32, 4),
                           transforms.ToTensor(),
                           normalize]
    else:
        transforms_list = [transforms.ToTensor(),
                           normalize]
    train_dset = dset.CIFAR10('data',
                              train=True,
                              download=True,
                              transform=transforms.Compose(transforms_list))
    test_dset = dset.CIFAR10('data',
                             train=False,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 normalize
                             ]))

    trainLoader = torch.utils.data.DataLoader(train_dset, batch_size=train_batch_size,
                                              shuffle=True, pin_memory=True)

    testLoader = torch.utils.data.DataLoader(test_dset, batch_size=test_batch_size,
                                             shuffle=False, pin_memory=True)

    return trainLoader, testLoader


def svhn_loaders(train_batch_size, test_batch_size=None):
    if test_batch_size is None:
        test_batch_size = train_batch_size

    normalize = transforms.Normalize(mean=[0.4377, 0.4438, 0.4728],
                                     std=[0.1980, 0.2010, 0.1970])
    train_loader = torch.utils.data.DataLoader(
        dset.SVHN(
            root='data', split='train', download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize
            ]),
        ),
        batch_size=train_batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        dset.SVHN(
            root='data', split='test', download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize
            ])),
        batch_size=test_batch_size, shuffle=False)
    return train_loader, test_loader


def expand_args(defaults, kwargs):
    d = defaults.copy()
    for k, v in kwargs.items():
        d[k] = v
    return d


MON_DEFAULTS = {
    'alpha': 1.0,
    'tol': 1e-5,
    'max_iter': 50
}


class SingleFcNet(nn.Module):

    def __init__(self, splittingMethod, in_dim=784, out_dim=100, m=0.1, **kwargs):
        super().__init__()
        linear_module = mon.MONSingleFc(in_dim, out_dim, m=m)
        nonlin_module = mon.MONReLU()
        self.mon = splittingMethod(
            linear_module, nonlin_module, **expand_args(MON_DEFAULTS, kwargs))
        self.Wout = nn.Linear(out_dim, 10)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        z = self.mon(x)
        return self.Wout(z[-1])


class NODENFcNet(nn.Module):

    def __init__(self, splittingMethod, in_dim=784, out_dim=100, m=0.1, **kwargs):
        super().__init__()
        linear_module = NODEN.NODEN_SingleFc(in_dim, out_dim, m=m)
        nonlin_module = NODEN.NODEN_ReLU()
        self.mon = splittingMethod(
            linear_module, nonlin_module, **expand_args(MON_DEFAULTS, kwargs))
        self.Wout = nn.Linear(out_dim, 10)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        z = self.mon(x)
        return self.Wout(z[-1])


class NODEN_Lip_Net(nn.Module):

    def __init__(self, splittingMethod, gamma, in_dim=784, width=100, out_dim=10, m=0.1, **kwargs):
        super().__init__()
        linear_module = NODEN.NODEN_Lipschitz_Fc(
            in_dim, width, out_dim, gamma, m=m)
        nonlin_module = NODEN.NODEN_ReLU()
        self.mon = splittingMethod(
            linear_module, nonlin_module, **expand_args(MON_DEFAULTS, kwargs))

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        z = self.mon(x)
        y = self.mon.linear_module.G(z[-1])
        return y


class MON_Lip_Net(nn.Module):

    def __init__(self, splittingMethod, gamma, in_dim=784, width=100, out_dim=10, m=0.1, **kwargs):
        super().__init__()
        linear_module = NODEN.Lipschitz_mon(in_dim, width, out_dim, gamma, m=m)
        nonlin_module = NODEN.NODEN_ReLU()
        self.mon = splittingMethod(
            linear_module, nonlin_module, **expand_args(MON_DEFAULTS, kwargs))

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        z = self.mon(x)
        y = self.mon.linear_module.G(z[-1])
        return y


class NODENFcNet_uncon(nn.Module):

    def __init__(self, splittingMethod, in_dim=784, out_dim=100, m=0.1, **kwargs):
        super().__init__()
        linear_module = NODEN.NODEN_SingleFc_uncon(in_dim, out_dim, m=m)
        nonlin_module = NODEN.NODEN_ReLU()
        self.mon = splittingMethod(
            linear_module, nonlin_module, **expand_args(MON_DEFAULTS, kwargs))
        self.Wout = nn.Linear(out_dim, 10)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        z = self.mon(x)
        return self.Wout(z[-1])


class NodenConvNet(nn.Module):

    def __init__(self, splittingMethod, in_dim=28, in_channels=1, out_channels=32, m=0.1, **kwargs):
        super().__init__()
        n = in_dim + 2
        shp = (n, n)
        self.pool = 4
        # self.out_dim = out_channels * (n // self.pool) ** 2
        self.out_dim = out_channels * (n) ** 2
        linear_module = NODEN.NODEN_Conv(
            in_dim, in_channels, out_channels, shp, m=m)
        nonlin_module = mon.MONBorderReLU(linear_module.pad[0])
        self.mon = splittingMethod(
            linear_module, nonlin_module, **expand_args(MON_DEFAULTS, kwargs))
        self.Wout = nn.Linear(self.out_dim, 10)

    def forward(self, x):
        x = F.pad(x, (1, 1, 1, 1))
        z = self.mon(x)
        z = F.avg_pool2d(z[-1], self.pool)
        return self.Wout(z.view(z.shape[0], -1))


class Noden_LipschitzConvNet(nn.Module):

    def __init__(self, splittingMethod, in_dim=28, in_channels=1, out_channels=32, m=0.1, gamma=1.0, pool=1, **kwargs):
        super().__init__()
        n = in_dim + 2
        shp = (n, n)
        self.pool = pool
        self.out_dim = out_channels * (n // self.pool) ** 2
        # self.out_dim = out_channels * (n) ** 2

        linear_module = NODEN.NODEN_Lipschitz_Conv(
            in_dim, in_channels, out_channels, self.out_dim, gamma, shp, m=m, pool=self.pool)
        nonlin_module = mon.MONBorderReLU(linear_module.pad[0])
        # nonlin_module = mon.MONBorderLin(linear_module.pad[0])

        self.mon = splittingMethod(
            linear_module, nonlin_module, **expand_args(MON_DEFAULTS, kwargs))

    def forward(self, x):
        x = F.pad(x, (1, 1, 1, 1))
        z = self.mon(x)
        z = F.avg_pool2d(z[-1], self.pool)
        return self.mon.linear_module.Wout(z.view(z.shape[0], -1))


class SingleConvNet(nn.Module):

    def __init__(self, splittingMethod, in_dim=28, in_channels=1, out_channels=32, m=0.1, **kwargs):
        super().__init__()
        n = in_dim + 2
        shp = (n, n)
        self.pool = 1
        self.out_dim = out_channels * (n // self.pool) ** 2
        linear_module = mon.MONSingleConv(in_channels, out_channels, shp, m=m)
        nonlin_module = mon.MONBorderReLU(linear_module.pad[0])
        self.mon = splittingMethod(
            linear_module, nonlin_module, **expand_args(MON_DEFAULTS, kwargs))
        self.Wout = nn.Linear(self.out_dim, 10)

    def forward(self, x):
        x = F.pad(x, (1, 1, 1, 1))
        z = self.mon(x)
        z = F.avg_pool2d(z[-1], self.pool)
        return self.Wout(z.view(z.shape[0], -1))


class UnconConvNet(nn.Module):

    def __init__(self, splittingMethod, in_dim=28, in_channels=1, out_channels=32, m=0.1, **kwargs):
        super().__init__()
        n = in_dim + 2
        shp = (n, n)
        self.pool = 4
        self.out_dim = out_channels * (n // self.pool) ** 2
        linear_module = NODEN.Uncon_Conv(in_channels, out_channels, shp, m=m)
        nonlin_module = mon.MONBorderReLU(linear_module.pad[0])
        self.mon = splittingMethod(
            linear_module, nonlin_module, **expand_args(MON_DEFAULTS, kwargs))
        self.Wout = nn.Linear(self.out_dim, 10)

    def forward(self, x):
        x = F.pad(x, (1, 1, 1, 1))
        z = self.mon(x)
        z = F.avg_pool2d(z[-1], self.pool)
        return self.Wout(z.view(z.shape[0], -1))

    # def forward(self, x):
    #     batch = 1
    #     channel = 1

    #     x = F.pad(x, (1, 1, 1, 1))
    #     z = self.mon(x)
    #     Mx = F.avg_pool2d(z[-1], self.pool)

    #     # MTMx = utils.avg_pool_adjoint(Mx)
    #     upsampler = torch.nn.Upsample(scale_factor=4)
    #     MTMx = utils.cpad(upsampler(Mx)) / 16

    #     # Compare innner product
    #     xTMTMx1 = z[-1][batch, channel].reshape(-1, 1).T @ MTMx[batch, channel].reshape(-1, 1)
    #     xTMTMx2 = Mx[batch, channel].reshape(-1, 1).T @ Mx[batch, channel].reshape(-1, 1)

    #     print("(xT) (MTMx)", xTMTMx1)
    #     print("(xTMT)(Mx))", xTMTMx2)

    #     return self.Wout(z.view(z.shape[0], -1))


class MultiConvNet(nn.Module):
    def __init__(self, splittingMethod, in_dim=28, in_channels=1,
                 conv_sizes=(16, 32, 64), m=1.0, **kwargs):
        super().__init__()
        linear_module = mon.MONMultiConv(
            in_channels, conv_sizes, in_dim+2, kernel_size=3, m=m)
        nonlin_module = mon.MONBorderReLU(linear_module.pad[0])
        self.mon = splittingMethod(
            linear_module, nonlin_module, **expand_args(MON_DEFAULTS, kwargs))
        out_shape = linear_module.z_shape(1)[-1]
        dim = out_shape[1]*out_shape[2]*out_shape[3]
        self.Wout = nn.Linear(dim, 10)

    def forward(self, x):
        x = F.pad(x, (1, 1, 1, 1))
        zs = self.mon(x)
        z = zs[-1]
        z = z.view(z.shape[0], -1)
        return self.Wout(z)


def test_robustness(model, testLoader, device='cuda', check_Lipschitz=True, Lip_batches=50):

    channels = 3
    dim = 32
    maxIter = 1000
    model = model.eval()
    # Lip_batches = 50  # Number of points to use when calculating the LC

    # Test nominal performance.
    test_loss = 0
    v_loss = 0
    incorrect_val = 0

    with torch.no_grad():
        for batch in testLoader:
            data, target = cuda(batch[0]), cuda(batch[1])
            preds = model(data)
            ce_loss = nn.CrossEntropyLoss(reduction='sum')(preds, target)
            test_loss += ce_loss
            v_loss += ce_loss.item()
            incorrect_val += preds.float().argmax(1).ne(target.data).sum()
        test_loss /= len(testLoader.dataset)
        nTotal = len(testLoader.dataset)
        err = 100. * incorrect_val.float() / float(nTotal)
        print('Test set: Average loss: {:.4f}, Error: {}/{} ({:.2f}%)'.format(
            test_loss, incorrect_val, nTotal, err))

    nominal_perf = err

    # Estimate Lipschitz constant of model
    if check_Lipschitz:
        Lip = 0
        u = torch.randn((Lip_batches, channels, dim, dim),
                        requires_grad=True, device=device)
        v = torch.randn_like(u, requires_grad=True, device=device)

        optimizer = torch.optim.Adam([u, v], lr=1E-2)
        iter = 0
        while iter < maxIter:
            optimizer.zero_grad()
            y1 = model(u)
            y2 = model(u + v)

            Lip_last = Lip
            Lip = (y1 - y2).norm(dim=1) ** 2 / \
                v.view([v.shape[0], -1]).norm(dim=1) ** 2
            Obj = -Lip.sum()
            Obj.backward()
            optimizer.step()

            print('\rLipschitz constant: {:1.3f}'.format(
                Lip.max().sqrt().item()), sep=' ', end='', flush=True)

            iter += 1
            if iter > 25:
                if Lip.max() < Lip_last.max() + 1E-4:  # Smaller than 1E-4 round-off error?
                    optimizer.param_groups[0]["lr"] /= 10.0
                    iter = 0

                    if optimizer.param_groups[0]["lr"] <= 1E-5:
                        break
        print()
        print()

    # Estimate Adversarial perturbations
    epsilons = np.linspace(1E-2, 5, 50)
    u = cuda(batch[0])
    v = torch.randn_like(u, requires_grad=True, device=device)
    v.data /= 100

    epsilons = np.linspace(1E-2, 15, 40)
    errors = []

    # Perform adversarial attacks
    bound_a = -0.1307 / 0.3081  # Calc. bound on pixel val due to normalization
    bound_b = (1 - 0.1307) / 0.3081
    fmodel = fb.PyTorchModel(model, bounds=(bound_a, bound_b))
    # attack = fb.attacks.L2ProjectedGradientDescentAttack()
    # attack = fb.attacks.L2DeepFoolAttack()
    attack = fb.attacks.L2FastGradientAttack()
    # attack = fb.attacks.L2CarliniWagnerAttack()
    raw, advs, success = attack(fmodel, u, target, epsilons=epsilons)

    errors = success.sum(dim=1).to('cpu').numpy() / float(batch[0].shape[0])

    results = {"nominal": nominal_perf.to('cpu').item(
    ), "epsilon": epsilons, "errors": errors, "Lipschitz": Lip.max().sqrt().item()}

    return results


def estimate_Lip(model, Lip_batches, channels, dim):   # Estimate Lipschitz constant of model

    device = "cuda" if torch.cuda.is_available else "cpu"
    maxIter = 500
    Lip = 0
    u = torch.randn((Lip_batches, channels, dim, dim),
                    requires_grad=True, device=device)
    v = torch.randn_like(u, requires_grad=True, device=device)

    optimizer = torch.optim.Adam([u, v], lr=1E-1)
    iter = 0
    while iter < maxIter:
        optimizer.zero_grad()
        y1 = model(u)
        y2 = model(u + v)

        Lip_last = Lip
        Lip = (y1 - y2).norm(dim=1) ** 2 / \
            v.view([v.shape[0], -1]).norm(dim=1) ** 2
        Obj = -Lip.sum()
        Obj.backward()
        optimizer.step()

        print('\rLipschitz constant: {:1.3f}'.format(
            Lip.max().sqrt().item()), sep=' ', end='', flush=True)

        iter += 1
        if iter > 25:
            if Lip.max() < Lip_last.max() + 1E-3:  # Smaller than 1E-4 round-off error?
                optimizer.param_groups[0]["lr"] /= 10.0
                iter = 0

                if optimizer.param_groups[0]["lr"] <= 1E-5:
                    break
    print()
    print()
