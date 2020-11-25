
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

    load_models = False
    use_double = False
    dataset = "cifar"

    if use_double:
        torch.set_default_tensor_type(torch.DoubleTensor)

    if dataset == "mnist":
        trainLoader, testLoader = train.mnist_loaders(train_batch_size=200,
                                                      test_batch_size=200,
                                                      use_double=use_double)
        in_dim = 28
        in_channels = 1
        pool = 3
        data_stats = {"feature_size": (in_channels, in_dim, in_dim),
                      "mean": (0.1307,),
                      "std": (0.3081,)}

    elif dataset == "cifar":
        trainLoader, testLoader = train.cifar_loaders(train_batch_size=128,
                                                      test_batch_size=500,
                                                      augment=False,
                                                      use_double=use_double)

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
    epochs = 25
    seed = 1
    tol = 1E-2
    # width = 81
    lr_decay_steps = 15

    max_iter = 200
    m = 0.1

    path = './models/conv_experiment_v3/'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for width in [40, 81, 120, 162, 200]:
        # # Train FF convolutional network
        name = 'ff_conv_w{:d}'.format(width)
        FFConvNet = train.FFConvNet(in_dim=in_dim,
                                    in_channels=in_channels,
                                    out_channels=width,
                                    pool=pool)

        if load_models:
            FFConvNet.load_state_dict(torch.load(path + name + '.params'))
        else:

            train_res, val_res = train.train(trainLoader, testLoader,
                                             FFConvNet,
                                             max_lr=1e-3,
                                             lr_mode='step',
                                             step=lr_decay_steps,
                                             change_mo=False,
                                             epochs=epochs,
                                             print_freq=100,
                                             tune_alpha=False,
                                             warmstart=False)

            name = 'ff_conv_w{:d}'.format(width)
            torch.save(FFConvNet.state_dict(), path + name + '.params')

        res = train.test_robustness(FFConvNet, testLoader, data_stats)
        # res["train"] = train_res
        # res["val"] = val_res
        io.savemat(path + name + ".mat", res)

    # for metric in ["full", "identity"]:
    for metric in ["full"]:
        alpha = 1.0
        max_alpha = 1.0

        torch.manual_seed(seed)
        numpy.random.seed(seed)

        # # # # Train and test Single layer convolutional LBEN
        name = metric + '_conv_w{:d}'.format(width)
        LbenConvNet = train.LBENConvNet(sp.MONForwardBackwardSplitting,
                                        in_dim=in_dim,
                                        in_channels=in_channels,
                                        out_channels=width,
                                        alpha=alpha,
                                        max_iter=max_iter,
                                        metric=metric,
                                        init="default",
                                        tol=tol,
                                        m=m,
                                        pool=pool,
                                        verbose=False)

        # Load or train a new model.
        if load_models:
            LbenConvNet.load_state_dict(torch.load(path + name + '.params'))

        else:
            train_res, val_res = train.train(trainLoader, testLoader,
                                             LbenConvNet,
                                             max_lr=1e-3,
                                             lr_mode='step',
                                             step=lr_decay_steps,
                                             change_mo=False,
                                             epochs=epochs,
                                             print_freq=100,
                                             tune_alpha=True,
                                             max_alpha=max_alpha,
                                             warmstart=False)

            torch.save(LbenConvNet.state_dict(), path + name + '.params')

        # Perform tests for nom performance, Lipschitz constant and robustness.
        LbenConvNet.mon.tol = 1E-3
        res = train.test_robustness(LbenConvNet, testLoader, data_stats)
        # res["train"] = train_res
        # res["val"] = val_res

        io.savemat(path + name + ".mat", res)

        # for gamma in [5.0, 50.0]:

        #     alpha = 0.5
        #     max_alpha = 1.0

        #     torch.manual_seed(seed)
        #     numpy.random.seed(seed)

        #     name = metric + '_conv_w{:d}_L{:1.1f}'.format(width, gamma)
        #     LipConvNet = train.LBENLipConvNetV2(sp.MONForwardBackwardSplitting,
        #                                         in_dim=in_dim,
        #                                         in_channels=in_channels,
        #                                         out_channels=width,
        #                                         alpha=alpha,
        #                                         max_iter=max_iter,
        #                                         metric=metric,
        #                                         init="default",
        #                                         tol=tol,
        #                                         m=m,
        #                                         gamma=gamma,
        #                                         pool=pool,
        #                                         verbose=False)

        #     if load_models:
        #         LipConvNet.load_state_dict(torch.load(path+name+'.params'))
        #     else:
        #         train_res, val_res = train.train(trainLoader, testLoader,
        #                                          LipConvNet,
        #                                          max_lr=1e-3,
        #                                          lr_mode='step',
        #                                          step=lr_decay_steps,
        #                                          change_mo=False,
        #                                          epochs=epochs,
        #                                          print_freq=100,
        #                                          tune_alpha=True,
        #                                          max_alpha=max_alpha,
        #                                          warmstart=False)

        #         torch.save(LipConvNet.state_dict(), path + name + '.params')

        #     LipConvNet.mon.tol = 1E-3
        #     res = train.test_robustness(LipConvNet, testLoader, data_stats)
        #     # res["train"] = train_res
        #     # res["val"] = val_res
        #     io.savemat(path + name + ".mat", res)

        #     print("fin")
