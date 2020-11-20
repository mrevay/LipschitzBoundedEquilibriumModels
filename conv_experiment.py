
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
    torch.set_default_tensor_type(torch.DoubleTensor)

    dataset = "cifar"
    if dataset == "mnist":
        trainLoader, testLoader = train.mnist_loaders(train_batch_size=200,
                                                      test_batch_size=200)
        in_dim = 28
        in_channels = 1
        pool = 3
        data_stats = {"feature_size": (in_channels, in_dim, in_dim),
                      "mean": (0.1307,),
                      "std": (0.3081,)}

    elif dataset == "cifar":
        trainLoader, testLoader = train.cifar_loaders(train_batch_size=128,
                                                      test_batch_size=250,
                                                      augment=False)

        in_dim = 32
        in_channels = 3
        pool = 2

        # stats used later by test_robustness
        data_stats = {"feature_size": (in_channels, in_dim, in_dim),
                      "mean": (0.4914, 0.4822, 0.4465),
                      "std": (0.2470, 0.2435, 0.2616)}

    elif dataset == "svhn":
        trainLoader, testLoader = train.svhn_loaders(train_batch_size=100,
                                                     test_batch_size=100)
        in_dim = 32
        in_channels = 3
        pool = 2

        data_stats = {"feature_size": (in_channels, in_dim, in_dim),
                      "mean": (0.4377, 0.4438, 0.4728),
                      "std": (0.1980, 0.2010, 0.1970)}

    # Choose between full, Identity, Channel, Image
    metric = "full"
    alpha = 0.5
    max_alpha = 0.5
    epochs = 25
    seed = 1
    tol = 1E-2
    # width = 81
    width = 81
    lr_decay_steps = 15
    max_iter = 250
    # m = 0.1
    m = 1.0

    path = './models/conv_experiment_v2/'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    torch.manual_seed(seed)
    numpy.random.seed(seed)

    # # Train and test Single layer convolutional LBEN
    # LbenConvNet = train.LBENConvNet(sp.MONForwardBackwardSplitting,
    #                                 in_dim=in_dim,
    #                                 in_channels=in_channels,
    #                                 out_channels=width,
    #                                 alpha=alpha,
    #                                 max_iter=max_iter,
    #                                 metric=metric,
    #                                 tol=tol,
    #                                 m=m,
    #                                 pool=pool,
    #                                 verbose=False)

    # train_res, val_res = train.train(trainLoader, testLoader,
    #                                  LbenConvNet,
    #                                  max_lr=1e-3,
    #                                  lr_mode='step',
    #                                  step=lr_decay_steps,
    #                                  change_mo=False,
    #                                  epochs=epochs,
    #                                  print_freq=100,
    #                                  tune_alpha=True,
    #                                  warmstart=False)

    # name = 'lben_conv_w{:d}'.format(width)
    # torch.save(LbenConvNet.state_dict(), path + name + '.params')

    # LbenConvNet.mon.tol = 1E-3
    # res = train.test_robustness(LbenConvNet, testLoader, data_stats)
    # res["train"] = train_res
    # res["val"] = val_res
    # io.savemat(path + name + ".mat", res)

    # # Compare with unbounded Single Conv Net
    # ConvNet = train.SingleConvNet(sp.MONForwardBackwardSplitting,
    #                               in_dim=in_dim,
    #                               in_channels=in_channels,
    #                               out_channels=width,
    #                               alpha=alpha,
    #                               max_iter=max_iter,
    #                               tol=tol,
    #                               m=m,
    #                               pool=pool,
    #                               verbose=True)

    # train_res, val_res = train.train(trainLoader, testLoader,
    #                                  ConvNet,
    #                                  max_lr=1e-3,
    #                                  lr_mode='step',
    #                                  step=lr_decay_steps,
    #                                  change_mo=False,
    #                                  epochs=epochs,
    #                                  print_freq=100,
    #                                  tune_alpha=True,
    #                                  warmstart=False)

    # name = 'mon_conv_w{:d}'.format(width)
    # torch.save(ConvNet.state_dict(), path + name + '.params')

    # ConvNet.mon.tol = 1E-3
    # res = train.test_robustness(ConvNet, testLoader, data_stats)
    # res["train"] = train_res
    # res["val"] = val_res
    # io.savemat(path + name + ".mat", res)

    # Lipschitz network
    # for gamma in [30.0, 10, 8.0, 5.0, 3.0, 0.8, 0.5, 0.3, 0.2]:

    for gamma in [5.0, 3.0]:

        torch.manual_seed(seed)
        numpy.random.seed(seed)

        LipConvNet = train.LBENLipConvNet(sp.MONForwardBackwardSplitting,
                                          in_dim=in_dim,
                                          in_channels=in_channels,
                                          out_channels=width,
                                          alpha=alpha,
                                          max_iter=max_iter,
                                          tol=tol,
                                          m=m,
                                          gamma=gamma,
                                          pool=pool,
                                          verbose=True)

        train_res, val_res = train.train(trainLoader, testLoader,
                                         LipConvNet,
                                         max_lr=1e-3,
                                         lr_mode='step',
                                         step=lr_decay_steps,
                                         change_mo=False,
                                         epochs=epochs,
                                         print_freq=100,
                                         tune_alpha=True,
                                         max_alpha=max_alpha,
                                         warmstart=False)

        name = 'lben_conv_w{:d}_L{:1.1f}'.format(width, gamma)
        torch.save(LipConvNet.state_dict(), path + name + '.params')

        LipConvNet.mon.tol = 1E-4
        res = train.test_robustness(LipConvNet, testLoader, data_stats)
        res["train"] = train_res
        res["val"] = val_res
        io.savemat(path + name + ".mat", res)

        print("fin")
