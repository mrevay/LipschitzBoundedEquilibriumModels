
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

    dataset = "svhn"
    if dataset == "mnist":
        trainLoader, testLoader = train.mnist_loaders(train_batch_size=1000,
                                                      test_batch_size=1000)
        in_dim = 28
        in_channels = 1
    elif dataset == "cifar":
        trainLoader, testLoader = train.cifar_loaders(train_batch_size=250,
                                                      test_batch_size=250)
        in_dim = 32
        in_channels = 3

    elif dataset == "svhn":
        trainLoader, testLoader = train.svhn_loaders(train_batch_size=250,
                                                     test_batch_size=250)
        in_dim = 32
        in_channels = 3

    load_models = False
    alpha = 0.005
    gamma = 0.5
    epochs = 10
    seed = 4
    tol = 1E-2
    width = 40
    lr_decay_steps = 5
    max_iter = 1500
    m = 10

    path = './models/conv_experiment/'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    torch.manual_seed(seed)
    numpy.random.seed(seed)

    # Lipschitz network
    for gamma in [0.5, 1.0, 2.0, 3.0, 5.0]:
        torch.manual_seed(seed)
        numpy.random.seed(seed)

        LipConvNet = train.Noden_LipschitzConvNet(sp.Broyden,
                                                  in_dim=in_dim,
                                                  in_channels=in_channels,
                                                  out_channels=width,
                                                  alpha=alpha,
                                                  max_iter=max_iter,
                                                  tol=tol,
                                                  m=m,
                                                  gamma=gamma,
                                                  pool=4)

        Lip_train, Lip_val = train.train(trainLoader, testLoader,
                                         LipConvNet,
                                         max_lr=1e-3,
                                         lr_mode='step',
                                         step=lr_decay_steps,
                                         change_mo=False,
                                         epochs=epochs,
                                         print_freq=100,
                                         tune_alpha=False)

        name = 'Lip_conv_w{:d}_L{:1.1f}'.format(width, gamma)
        torch.save(LipConvNet.state_dict(), path + name + '.params')

        LipConvNet.mon.tol = 1E-4
        res = train.test_robustness(LipConvNet, testLoader)
        io.savemat(path + name + ".mat", res)

    print("fin")
