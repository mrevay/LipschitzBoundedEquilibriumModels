
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

    trainLoader, testLoader = train.mnist_loaders(train_batch_size=128,
                                                  test_batch_size=400)

    # trainLoader, testLoader = train.cifar_loaders(train_batch_size=128, test_batch_size=400)
    # trainLoader, testLoader = train.cifar_loaders(train_batch_size=128, test_batch_size=400, augment=False)

    epochs = 80
    seed = 8
    tol = 1E-3
    width = 80
    lr_decay_steps = 20

    image_size = 28 * 28

    # Ode stability condition
    torch.manual_seed(seed)
    numpy.random.seed(seed)

    LipNet = train.NODEN_Lip_Net(sp.MONPeacemanRachford,
                                 in_dim=image_size,
                                 width=width,
                                 out_dim=10,
                                 alpha=1.0,
                                 max_iter=300,
                                 tol=tol,
                                 m=1.0,
                                 gamma=10)

    lip_train, lip_test = train.train(trainLoader, testLoader,
                                      LipNet,
                                      max_lr=1e-3,
                                      lr_mode='step',
                                      step=lr_decay_steps,
                                      change_mo=False,
                                       epochs=epochs,
                                      print_freq=100,
                                      tune_alpha=True)




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

    # X = torch.randn(100, 3*32*32)
    # Z = torch.randn(100, width)

    # mon = monNet.mon.linear_module
    # ode = odeNet.mon.linear_module

    # ode.init_inverse(1, -1)
    # mon.init_inverse(1, -1)

    # print("multiply is the same: ", (mon.multiply(Z)[0] - ode.multiply(Z)[0]).norm().item() < 1E-3)
    # print("multiply_tranpose is the same: ", (mon.multiply_transpose(Z)[0] - ode.multiply_transpose(Z)[0]).norm().item() < 1E-3)
    # print("inverse is the same: ", (mon.inverse(Z)[0] - ode.inverse(Z)[0]).norm().item() < 1E-3)
    # print("inverse transpose is the same: ", (mon.inverse_transpose(Z)[0] - ode.inverse_transpose(Z)[0]).norm().item() < 1E-3)



    p1, = plt.plot(ode_train)
    p2, = plt.plot(mon_train)
    p3, = plt.plot(uncon_train)
    plt.legend([p1, p2, p3], ["ode", "mon", "uncon"])
    # plt.yscale('log')
    plt.show()

    print('~fin~')