
import matplotlib

import splitting as sp
import train

import torch
import numpy

import NODEN
import mon

import scipy.io as io
import matplotlib.pyplot as plt

if __name__ == "__main__":

    trainLoader, testLoader = train.mnist_loaders(train_batch_size=128,
                                                  test_batch_size=4000)

    load_models = False

    path = './models/mnist_capacity/'
    epochs = 30
    seed = 2
    tol = 1E-3  # Turn up tolerance when concerned about Lipschitz bound.
    width = 10
    lr_decay_steps = 10

    image_size = 28 * 28

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Ode stability condition
    torch.manual_seed(seed)
    numpy.random.seed(seed)
    name = 'ode_w{:d}'.format(width)

    odeNet = train.NODENFcNet(sp.MONPeacemanRachford,
                              in_dim=image_size,
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

    torch.save(odeNet.state_dict(), path + name + '.params')

    results = {"train": ode_train, "test": ode_test}
    io.savemat(path + 'ode.mat', results)

    # Monotone operator network
    torch.manual_seed(seed)
    numpy.random.seed(seed)
    name = 'mon_w{:d}'.format(width)

    monNet = train.SingleFcNet(sp.MONPeacemanRachford,
                               in_dim=image_size,
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
    torch.save(monNet.state_dict(), path + name + '.params')

    results = {"train": mon_train, "test": mon_test}
    io.savemat(path + 'mon.mat', results)

    # unconstrained network
    torch.manual_seed(seed)
    numpy.random.seed(seed)

    name = 'uncon_w{:d}'.format(width)

    unconNet = train.NODENFcNet_uncon(sp.MONPeacemanRachford,
                                      in_dim=image_size,
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
    torch.save(unconNet.state_dict(), path + name + '.params')

    results = {"train": uncon_train, "test": uncon_test}
    io.savemat(path + 'uncon.mat', results)

    print('~fin~')
