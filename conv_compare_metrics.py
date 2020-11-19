
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
    # torch.set_default_tensor_type(torch.DoubleTensor)

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
                                                      test_batch_size=128,
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

    alpha = 1.0
    epochs = 40
    seed = 1
    tol = 1E-2
    width = 81
    lr_decay_steps = 25
    max_iter = 250
    m = 0.1

    path = './models/conv_compare_metrics/' + dataset + "/"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # # Train and test Single layer convolutional LBEN
    for metric in ["identity", "full", "channels", "image"]:
        torch.manual_seed(seed)
        numpy.random.seed(seed)

        name = metric + '_w{:d}'.format(width)
        print("Training model ", name)

        LbenConvNet = train.LBENConvNet(sp.MONForwardBackwardSplitting,
                                        in_dim=in_dim,
                                        in_channels=in_channels,
                                        out_channels=width,
                                        alpha=alpha,
                                        max_iter=max_iter,
                                        metric=metric,
                                        tol=tol,
                                        m=m,
                                        pool=pool,
                                        verbose=False)

        train_res, val_res = train.train(trainLoader, testLoader,
                                         LbenConvNet,
                                         max_lr=1e-3,
                                         lr_mode='step',
                                         step=lr_decay_steps,
                                         change_mo=False,
                                         epochs=epochs,
                                         print_freq=100,
                                         tune_alpha=True,
                                         warmstart=False)

        print("Done! Saving model at location ", path + name)
        torch.save(LbenConvNet.state_dict(), path + name + '.params')

        LbenConvNet.mon.tol = 1E-3
        res = train.test_robustness(LbenConvNet, testLoader, data_stats)
        res["train"] = train_res
        res["val"] = val_res
        io.savemat(path + name + ".mat", res)
