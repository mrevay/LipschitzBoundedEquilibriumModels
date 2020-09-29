import matplotlib
matplotlib.use("TkAgg")

import splitting as sp
import train

import torch
import numpy

import NODEN
import mon

import scipy.io as io

import matplotlib.pyplot as plt

# torch.set_default_dtype(torch.float64)  # double precision for SDP
# torch.set_printoptions(precision=4)

if __name__ == "__main__":

    # trainLoader, testLoader = train.mnist_loaders(train_batch_size=128,
    #                                               test_batch_size=400)

    # trainLoader, testLoader = train.cifar_loaders(train_batch_size=128, test_batch_size=400)
    trainLoader, testLoader = train.cifar_loaders(train_batch_size=128, test_batch_size=400, augment=False)

    epochs = 50
    seed = 7
    tol = 1E-3
    width = 100
    lr_decay_steps = 20

    torch.manual_seed(seed)
    numpy.random.seed(seed)

    # unconstraIEND NETWORK
    unconNet = train.NODENFcNet_uncon(sp.MONPeacemanRachford,
                   in_dim=3*32*32,
                   out_dim=width,
                   alpha=1.0,
                   max_iter=300,
                   tol=tol,
                   m=1.0)


    uncon_train, uncon_test = train.train(trainLoader, testLoader,
                unconNet,
                max_lr=1e-3,
                lr_mode='step',
                step=lr_decay_steps,
                change_mo=False,
                epochs=epochs,    
                print_freq=100,
                tune_alpha=True)


    # Ode stability condition
    torch.manual_seed(seed)
    numpy.random.seed(seed)

    odeNet = train.NODENFcNet(sp.MONPeacemanRachford,
                   in_dim=3*32*32,
                   out_dim=width,
                   alpha=1.0,
                   max_iter=300,
                   tol=tol,
                   m=1.0)


    ode_train, ode_test = train.train(trainLoader, testLoader,
                odeNet,
                max_lr=1e-3,
                lr_mode='step',
                step=lr_decay_steps,
                change_mo=False,
                epochs=epochs,    
                print_freq=100,
                tune_alpha=True)


    # Monotone operator network
    torch.manual_seed(seed)
    numpy.random.seed(seed)

    monNet = train.SingleFcNet(sp.MONPeacemanRachford,
                              in_dim=3*32*32,
                              out_dim=width,
                              alpha=1.0,
                              max_iter=300,
                              tol=tol,
                              m=1.0)

    mon_train, mon_test = train.train(trainLoader, testLoader,
                monNet,
                max_lr=1e-3,
                lr_mode='step',
                step=lr_decay_steps,
                change_mo=False,
                epochs=epochs,
                print_freq=100,
                tune_alpha=True)