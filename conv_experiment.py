
import matplotlib
matplotlib.use("TkAgg")

import splitting as sp
import train

import torch
import numpy

import NODEN
import mon

import scipy.io as io
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":

    dataset = "mnist"
    if dataset == "mnist":
        trainLoader, testLoader = train.mnist_loaders(train_batch_size=128,
                                                      test_batch_size=128)
        in_dim = 28
        in_channels = 1
    elif dataset == "cifar":
        trainLoader, testLoader = train.cifar_loaders(train_batch_size=256,
                                                      test_batch_size=256)
        in_dim = 32
        in_channels = 3

    gamma = 0.3
    epochs = 25
    seed = 4
    tol = 1E-3
    width = 20
    lr_decay_steps = 10

    torch.manual_seed(seed)
    numpy.random.seed(seed)

    # # Test unconstrained convnet
    # unconNet = train.UnconConvNet(sp.FISTA,
    #                               in_dim=in_dim,
    #                               in_channels=in_channels,
    #                               out_channels=width,
    #                               alpha=0.5,
    #                               max_iter=300,
    #                               tol=1e-3,
    #                               m=1.0)

    # uncon_train, uncon_val = train.train(trainLoader, testLoader,
    #                                      unconNet,
    #                                      max_lr=1e-3,
    #                                      lr_mode='step',
    #                                      step=lr_decay_steps,
    #                                      change_mo=False,
    #                                      epochs=epochs,
    #                                      print_freq=100,
    #                                      tune_alpha=False)

    path = './models/conv_experiment/'
    # name = 'uncon_conv_w{:d}'.format(width)
    # torch.save(unconNet.state_dict(), path + name + '.params')

    odeConvNet = train.NodenConvNet(sp.FISTA,
                                    in_dim=in_dim,
                                    in_channels=in_channels,
                                    out_channels=width,
                                    alpha=1.0,
                                    max_iter=300,
                                    tol=1e-3,
                                    m=1.0)

    X = torch.randn((1, 30 , 30))
    pool = 4

    def pooling(x): 
        return torch.nn.functional.avg_pool2d(x, pool)

    def cpad(x):
        return torch.nn.functional.pad(x, 4*(1,), mode="circular")

    def pool_adjoint(x):
        'Calculate the adjoint of average pooling operator'
        adj = torch.nn.functional.upsample(x, scale_factor=pool) / pool ** 2
        return adj

    Px = pooling(X)
    PTPx = pool_adjoint(Px)

    xPTPx1 = Px.T @ Px
    xPTPx2 = X.T @ PTPx

    print("(Px)* (Px): ", xPTPx1, 'x* (P*Px):', xPTPx2)


    torch.manual_seed(seed)
    numpy.random.seed(seed)

    # Test Zico and Winston convnet
    convNet = train.SingleConvNet(sp.FISTA,
                                  in_dim=in_dim,
                                  in_channels=in_channels,
                                  out_channels=width,
                                  alpha=1.0,
                                  max_iter=300,
                                  tol=1e-3,
                                  m=1.0)


    mon_train, mon_val = train.train(trainLoader, testLoader,
                                     convNet,
                                     max_lr=1e-3,
                                     lr_mode='step',
                                     step=lr_decay_steps,
                                     change_mo=False,
                                     epochs=epochs,
                                     print_freq=100,
                                     tune_alpha=False)

    name = 'mon_conv_w{:d}'.format(width)
    torch.save(convNet.state_dict(), path + name + '.params')

    # Our network
    torch.manual_seed(seed)
    numpy.random.seed(seed)

    odeConvNet = train.NodenConvNet(sp.FISTA,
                                    in_dim=in_dim,
                                    in_channels=in_channels,
                                    out_channels=width,
                                    alpha=1.0,
                                    max_iter=300,
                                    tol=1e-3,
                                    m=1.0)

    ode_train, ode_val = train.train(trainLoader, testLoader,
                                     odeConvNet,
                                     max_lr=1e-3,
                                     lr_mode='step',
                                     step=lr_decay_steps,
                                     change_mo=False,
                                     epochs=epochs,
                                     print_freq=100,
                                     tune_alpha=False)

    name = 'ode_conv_w{:d}'.format(width)
    torch.save(odeConvNet.state_dict(), path + name + '.params')




    # # Lipschitz network
    # torch.manual_seed(seed)
    # numpy.random.seed(seed)

    # odeConvNet = train.Noden_LipschitzConvNet(sp.MONForwardBackwardSplitting,
    #                                           in_dim=in_dim,
    #                                           in_channels=in_channels,
    #                                           out_channels=width,
    #                                           alpha=0.5,
    #                                           max_iter=600,
    #                                           tol=1e-3,
    #                                           m=1,
    #                                           gamma=gamma)

    # ode_train, ode_val = train.train(trainLoader, testLoader,
    #                                  odeConvNet,
    #                                  max_lr=1e-3,
    #                                  lr_mode='step',
    #                                  step=lr_decay_steps,
    #                                  change_mo=False,
    #                                  epochs=epochs,
    #                                  print_freq=100,
    #                                  tune_alpha=True)

    # path = './models/'
    # name = 'lipschitz{:1.1f}_conv_w{:d}'.format(gamma, width)
    # torch.save(odeConvNet.state_dict(), path + name + '.params')

    # res = train.test_robustness(lmt0, testLoader)
    # io.savemat(path + name + '.mat', res)