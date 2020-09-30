
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as io
import mon
import NODEN
import numpy
import torch
import train
import splitting as sp
import matplotlib
matplotlib.use("TkAgg")


if __name__ == "__main__":

    dataset = "cifar"
    if dataset == "mnist":
        trainLoader, testLoader = train.mnist_loaders(train_batch_size=1000,
                                                      test_batch_size=1000)
        in_dim = 28
        in_channels = 1
    elif dataset == "cifar":
        trainLoader, testLoader = train.cifar_loaders(train_batch_size=128,
                                                      test_batch_size=128)
        in_dim = 32
        in_channels = 3

    alpha = 0.2
    gamma = 1.0
    epochs = 20
    seed = 4
    tol = 1E-4
    width = 81
    lr_decay_steps = 10
    max_iter = 1500

    torch.manual_seed(seed)
    numpy.random.seed(seed)

    path = './models/conv_experiment/'

    torch.manual_seed(seed)
    numpy.random.seed(seed)

    # Test Zico and Winston convnet
    convNet = train.SingleConvNet(sp.FISTA,
                                  in_dim=in_dim,
                                  in_channels=in_channels,
                                  out_channels=width,
                                  alpha=alpha,
                                  max_iter=max_iter,
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
    res = train.test_robustness(convNet, testLoader)
    io.savemat(path + name + ".mat", res)

    # LBEN Network
    # torch.manual_seed(seed)
    # numpy.random.seed(seed)

    # odeConvNet = train.NodenConvNet(sp.FISTA,
    #                                 in_dim=in_dim,
    #                                 in_channels=in_channels,
    #                                 out_channels=width,
    #                                 alpha=alpha,
    #                                 max_iter=max_iter,
    #                                 tol=tol,
    #                                 m=1.0)

    # ode_train, ode_val = train.train(trainLoader, testLoader,
    #                                  odeConvNet,
    #                                  max_lr=1e-3,
    #                                  lr_mode='step',
    #                                  step=lr_decay_steps,
    #                                  change_mo=False,
    #                                  epochs=epochs,
    #                                  print_freq=100,
    #                                  tune_alpha=False)

    # name = 'LBEN_conv_w{:d}'.format(width, gamma)
    # torch.save(odeConvNet.state_dict(), path + name + '.params')
    # res = train.test_robustness(odeConvNet, testLoader)
    # io.savemat(path + name + ".mat", res)

    # Lipschitz network
    torch.manual_seed(seed)
    numpy.random.seed(seed)

    LipConvNet = train.Noden_LipschitzConvNet(sp.FISTA,
                                              in_dim=in_dim,
                                              in_channels=in_channels,
                                              out_channels=width,
                                              alpha=alpha,
                                              max_iter=max_iter,
                                              tol=tol,
                                              m=1,
                                              gamma=gamma)

    Lip_train, Lip_val = train.train(trainLoader, testLoader,
                                     odeConvNet,
                                     max_lr=1e-3,
                                     lr_mode='step',
                                     step=lr_decay_steps,
                                     change_mo=False,
                                     epochs=epochs,
                                     print_freq=100,
                                     tune_alpha=False)

    name = 'Lip_conv_w{:d}_L{:1.1f}'.format(width, gamma)
    torch.save(LipConvNet.state_dict(), path + name + '.params')
    res = train.test_robustness(LipConvNet, testLoader)
    io.savemat(path + name + ".mat", res)

    print("fin")
