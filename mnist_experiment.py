
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
        x = x.view(x.shape[0], -1) * 0.3081 + 0.1307
        x = self.Win(x) 
        x = torch.relu(x)
        return self.Wout(x)


if __name__ == "__main__":

    trainLoader, testLoader = train.mnist_loaders(train_batch_size=512,
                                                  test_batch_size=512)

    load_models = False

    path = './models/'
    epochs = 40
    seed = 1
    tol = 1E-3
    width = 80
    lr_decay_steps = 20

    image_size = 28 * 28

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Lipschitz Networks
    models = []
    results = []
    for gamma in [0.2, 0.3, 0.4, 0.5, 0.8, 1.0, 5.0]:

        torch.manual_seed(seed)
        numpy.random.seed(seed)

        name = 'fc_lip{:2.1f}_w{:d}'.format(gamma, width)

        LipNet = train.NODEN_Lip_Net(sp.MONPeacemanRachford,
                                     in_dim=image_size,
                                     width=width,
                                     out_dim=10,
                                     alpha=1.0,
                                     max_iter=300,
                                     tol=tol,
                                     m=1.0,
                                     gamma=gamma)

        if load_models:
            LipNet.load_state_dict(torch.load(path + name + '.params'))
            LipNet.to(device)

        else:
            lip_train, lip_test = train.train(trainLoader, testLoader,
                                            LipNet,
                                            max_lr=1e-3,
                                            lr_mode='step',
                                            step=lr_decay_steps,
                                            change_mo=False,
                                            epochs=epochs,
                                            print_freq=100,
                                            tune_alpha=True)
            torch.save(LipNet.state_dict(), path + name + '.params')

        print('Testing model: ', name)
        res = train.test_robustness(LipNet, testLoader)
        io.savemat(path + name + '.mat', res)

        models += [LipNet]
        results += [res]


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
    if load_models:
        unconNet.load_state_dict(torch.load(path + name + '.params'))
        unconNet.to(device)

    else:
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

    res = train.test_robustness(unconNet, testLoader)
    io.savemat(path + name + '.mat', res)


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

    if load_models:
        odeNet.load_state_dict(torch.load(path + name + '.params'))
        odeNet.to(device)
    else:
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

    res = train.test_robustness(odeNet, testLoader)
    io.savemat(path + name + '.mat', res)


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
    if load_models:
        monNet.load_state_dict(torch.load(path + name + '.params'))
        monNet.to(device)

    else:
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

    res = train.test_robustness(monNet, testLoader)
    io.savemat(path + name + '.mat', res)

    name = 'lmt_c1_w{:d}'.format(width)
    lmt0 = simple_fc(image_size, width, 10, './models/lmt_models/mnist_weights_c1.0.mat')
    lmt0.cuda()
    print('Testing model: ', name)
    res = train.test_robustness(lmt0, testLoader)
    io.savemat(path + name + '.mat', res)

    name = 'lmt_c10_w{:d}'.format(width)
    lmt0 = simple_fc(image_size, width, 10, './models/lmt_models/mnist_weights_c10.0.mat')
    lmt0.cuda()
    print('Testing model: ', name)
    res = train.test_robustness(lmt0, testLoader)
    io.savemat(path + name + '.mat', res)

    name = 'lmt_c100_w{:d}'.format(width)
    lmt0 = simple_fc(image_size, width, 10, './models/lmt_models/mnist_weights_c100.0.mat')
    lmt0.cuda()
    print('Testing model: ', name)
    res = train.test_robustness(lmt0, testLoader)
    io.savemat(path + name + '.mat', res)

    name = 'lmt_c250_w{:d}'.format(width)
    lmt0 = simple_fc(image_size, width, 10, './models/lmt_models/mnist_weights_c250.0.mat')
    lmt0.cuda()
    print('Testing model: ', name)
    res = train.test_robustness(lmt0, testLoader)
    io.savemat(path + name + '.mat', res)

    name = 'lmt_c500_w{:d}'.format(width)
    lmt0 = simple_fc(image_size, width, 10, './models/lmt_models/mnist_weights_c500.0.mat')
    lmt0.cuda()
    print('Testing model: ', name)
    res = train.test_robustness(lmt0, testLoader)
    io.savemat(path + name + '.mat', res)

    name = 'lmt_c1000_w{:d}'.format(width)
    lmt0 = simple_fc(image_size, width, 10, './models/lmt_models/mnist_weights_c1000.0.mat')
    lmt0.cuda()
    print('Testing model: ', name)
    res = train.test_robustness(lmt0, testLoader)
    io.savemat(path + name + '.mat', res)

    print('~fin~')