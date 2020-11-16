
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as io
import mon
import NODEN
import numpy
import torch
import train
import splitting as sp
# import matplotlib
# matplotlib.use("TkAgg")


if __name__ == "__main__":

    dataset = "cifar"
    if dataset == "mnist":
        trainLoader, testLoader = train.mnist_loaders(train_batch_size=200,
                                                      test_batch_size=200)
        in_dim = 28
        in_channels = 1
        pool = 3
    elif dataset == "cifar":
        trainLoader, testLoader = train.cifar_loaders(train_batch_size=200,
                                                      test_batch_size=200)
        in_dim = 32
        in_channels = 3
        pool = 2

    elif dataset == "svhn":
        trainLoader, testLoader = train.svhn_loaders(train_batch_size=100,
                                                     test_batch_size=100)
        in_dim = 32
        in_channels = 3
        pool = 2

    load_models = False
    alpha = 0.05
    epochs = 60
    seed = 1
    tol = 1E-2
    width = 80
    lr_decay_steps = 5
    max_iter = 2000
    m = 1.0

    path = './models/conv_experiment_v2/'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    torch.manual_seed(seed)
    numpy.random.seed(seed)

    # Train and test Single layer convolutional LBEN
    LbenConvNet = train.LBENConvNet(sp.FISTA,
                                    in_dim=in_dim,
                                    in_channels=in_channels,
                                    out_channels=width,
                                    alpha=alpha,
                                    max_iter=max_iter,
                                    tol=tol,
                                    m=m)

    train, val = train.train(trainLoader, testLoader,
                             LbenConvNet,
                             max_lr=1e-3,
                             lr_mode='step',
                             step=lr_decay_steps,
                             change_mo=False,
                             epochs=epochs,
                             print_freq=100,
                             tune_alpha=False,
                             warmstart=False)

    name = 'lben_conv_w{:d}'.format(width)
    torch.save(LbenConvNet.state_dict(), path + name + '.params')

    LbenConvNet.mon.tol = 1E-3
    res = train.test_robustness(LbenConvNet, testLoader)
    res["train"] = train
    res["val"] = val
    io.savemat(path + name + ".mat", res)

    # Compare with unbounded Single Conv Net
    ConvNet = train.SingleConvNet(sp.FISTA,
                                  in_dim=in_dim,
                                  in_channels=in_channels,
                                  out_channels=width,
                                  alpha=alpha,
                                  max_iter=max_iter,
                                  tol=tol,
                                  m=m)

    train, val = train.train(trainLoader, testLoader,
                             ConvNet,
                             max_lr=1e-3,
                             lr_mode='step',
                             step=lr_decay_steps,
                             change_mo=False,
                             epochs=epochs,
                             print_freq=100,
                             tune_alpha=False,
                             warmstart=False)

    name = 'mon_conv_w{:d}'.format(width)
    torch.save(ConvNet.state_dict(), path + name + '.params')

    ConvNet.mon.tol = 1E-3
    res = train.test_robustness(ConvNet, testLoader)
    res["train"] = train
    res["val"] = val
    io.savemat(path + name + ".mat", res)

    # Lipschitz network
    for gamma in [0.5]:
        torch.manual_seed(seed)
        numpy.random.seed(seed)

        LipConvNet = train.LBENLipConvNet(sp.FISTA,
                                          in_dim=in_dim,
                                          in_channels=in_channels,
                                          out_channels=width,
                                          alpha=alpha,
                                          max_iter=max_iter,
                                          tol=tol,
                                          m=m,
                                          gamma=gamma,
                                          pool=pool)

        Lip_train, Lip_val = train.train(trainLoader, testLoader,
                                         LipConvNet,
                                         max_lr=1e-3,
                                         lr_mode='step',
                                         step=lr_decay_steps,
                                         change_mo=False,
                                         epochs=epochs,
                                         print_freq=100,
                                         tune_alpha=False,
                                         warmstart=False)

        name = 'conv_w{:d}_L{:1.1f}'.format(width, gamma)
        torch.save(LipConvNet.state_dict(), path + name + '.params')

        LipConvNet.mon.tol = 1E-3
        res = train.test_robustness(LipConvNet, testLoader)
        io.savemat(path + name + ".mat", res)

        print("fin")
