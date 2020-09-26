
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

    # trainLoader, testLoader = train.mnist_loaders(train_batch_size=128,
    #                                               test_batch_size=128)

    trainLoader, testLoader = train.cifar_loaders(train_batch_size=256,
                                                  test_batch_size=256)

    epochs = 30
    seed = 3
    tol = 1E-3
    width = 81
    lr_decay_steps = 10
    image_size = 32 * 32

    torch.manual_seed(seed)
    numpy.random.seed(seed)


    # Test Zico's convnet
    convNet = train.SingleConvNet(sp.MONForwardBackwardSplitting,
                                  in_dim=32,
                                  in_channels=3,
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
                                     tune_alpha=True)

    path = './models/'
    name = 'mon_conv_w{:d}'.format(width)
    torch.save(convNet.state_dict(), path + name + '.params')

    # Our network
    torch.manual_seed(seed)
    numpy.random.seed(seed)

    odeConvNet = train.NodenConvNet(sp.MONForwardBackwardSplitting,
                                    in_dim=32,
                                    in_channels=3,
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
                                     tune_alpha=True)

    path = './models/'
    name = 'ode_conv_w{:d}'.format(width)
    torch.save(odeConvNet.state_dict(), path + name + '.params')

    # Test robustness and nominal performance of the two models
    res = train.test_robustness(odeConvNet, testLoader)
    io.savemat(path + name + '.mat', res)

    res = train.test_robustness(convNet, testLoader)
    io.savemat(path + name + '.mat', res)
   