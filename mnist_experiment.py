
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


class simple_fc(torch.nn.Module):
    """ MON class with a single 3x3 (circular) convolution """

    def __init__(self, in_dim, width, out_dim, path=None):
        super().__init__()

        self.Win = torch.nn.Linear(in_dim, width)
        self.Wout = torch.nn.Linear(width, out_dim)

        if path is not None:
            data = io.loadmat(path)
            self.Win.weight.data = torch.Tensor(data["W1"])
            self.Win.bias.data = torch.Tensor(data["b1"][0])

            self.Wout.weight.data = torch.Tensor(data["W2"])
            self.Wout.bias.data = torch.Tensor(data["b2"][0])

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.Win(x) 
        x = torch.relu(x)
        return self.Wout(x)


if __name__ == "__main__":

    trainLoader, testLoader = train.mnist_loaders(train_batch_size=128,
                                                  test_batch_size=2000)

    path = './models/'
    epochs = 40
    seed = 8
    tol = 1E-3
    width = 80
    lr_decay_steps = 20

    image_size = 28 * 28


    # Lipschitz Networks
    models = []
    results = []
    for gamma in [0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 1.0, 5.0]:

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
                                    gamma=gamma)

        lip_train, lip_test = train.train(trainLoader, testLoader,
                                        LipNet,
                                        max_lr=1e-3,
                                        lr_mode='step',
                                        step=lr_decay_steps,
                                        change_mo=False,
                                        epochs=epochs,
                                        print_freq=100,
                                        tune_alpha=True)

        res = train.test_robustness(LipNet, testLoader)
       
        name = 'mon_lip{:2.1f}_w{:d}'.format(gamma, width)
        torch.save(LipNet.state_dict(), path + name + '.params')
        io.savemat(path + name + '.mat', res)

        models += [LipNet]
        results += [res]


    lmt0 = simple_fc(image_size, width, 10, './models/lmt_models/mnist_weights_c1.mat')
    lmt0.cuda()
    res = train.test_robustness(lmt0, testLoader)
    name = 'lmt_c1_w{:d}'.format(width)
    io.savemat(path + name + '.mat', res)

    lmt0 = simple_fc(image_size, width, 10, './models/lmt_models/mnist_weights_c10.0.mat')
    lmt0.cuda()
    res = train.test_robustness(lmt0, testLoader)
    name = 'lmt_c10_w{:d}'.format(width)
    io.savemat(path + name + '.mat', res)

    lmt0 = simple_fc(image_size, width, 10, './models/lmt_models/mnist_weights_c100.0.mat')
    lmt0.cuda()
    res = train.test_robustness(lmt0, testLoader)
    name = 'lmt_c100_w{:d}'.format(width)
    io.savemat(path + name + '.mat', res)

    lmt0 = simple_fc(image_size, width, 10, './models/lmt_models/mnist_weights_c250.0.mat')
    lmt0.cuda()
    res = train.test_robustness(lmt0, testLoader)
    name = 'lmt_c250_w{:d}'.format(width)
    io.savemat(path + name + '.mat', res)

    lmt0 = simple_fc(image_size, width, 10, './models/lmt_models/mnist_weights_c500.0.mat')
    lmt0.cuda()
    res = train.test_robustness(lmt0, testLoader)
    name = 'lmt_c500_w{:d}'.format(width)
    io.savemat(path + name + '.mat', res)

    lmt0 = simple_fc(image_size, width, 10, './models/lmt_models/mnist_weights_c1000.0.mat')
    lmt0.cuda()
    res = train.test_robustness(lmt0, testLoader)
    name = 'lmt_c1000_w{:d}'.format(width)
    io.savemat(path + name + '.mat', res)

    # Lipschitz Networks
    models = []
    results = []
    for gamma in [0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 1.0, 5.0]:

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
                                    gamma=gamma)

        lip_train, lip_test = train.train(trainLoader, testLoader,
                                        LipNet,
                                        max_lr=1e-3,
                                        lr_mode='step',
                                        step=lr_decay_steps,
                                        change_mo=False,
                                        epochs=epochs,
                                        print_freq=100,
                                        tune_alpha=True)

        res = train.test_robustness(LipNet, testLoader)
       
        name = 'fc_lip{:2.1f}_w{:d}'.format(gamma, width)
        torch.save(LipNet.state_dict(), path + name + '.params')
        io.savemat(path + name + '.mat', res)

        models += [LipNet]
        results += [res]


    # unconstrained network

    torch.manual_seed(seed)
    numpy.random.seed(seed)

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

    res = train.test_robustness(unconNet, testLoader)
    path = './models/'
    name = 'uncon_w{:d}'.format(width)
    torch.save(unconNet.state_dict(), path + name + '.params')
    io.savemat(path + name + '.mat', res)


    # Ode stability condition
    torch.manual_seed(seed)
    numpy.random.seed(seed)

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

    res = train.test_robustness(odeNet, testLoader)
    path = './models/'
    name = 'ode_w{:d}'.format(width)
    torch.save(unconNet.state_dict(), path + name + '.params')
    io.savemat(path + name + '.mat', res)

    # Monotone operator network
    torch.manual_seed(seed)
    numpy.random.seed(seed)

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

    res = train.test_robustness(odeNet, testLoader)
    path = './models/'
    name = 'mon_w{:d}'.format(width)
    torch.save(unconNet.state_dict(), path + name + '.params')
    io.savemat(path + name + '.mat', res)

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