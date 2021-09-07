import matplotlib.pyplot as plt
import scipy.io as io

if __name__ == '__main__':

    # fc02 = io.loadmat("./models/fc_lip0.2_w80.mat")
    # fc03 = io.loadmat("./models/fc_lip0.3_w80.mat")
    # fc04 = io.loadmat("./models/fc_lip0.4_w80.mat")
    # fc05 = io.loadmat("./models/fc_lip0.5_w80.mat")
    # fc08 = io.loadmat("./models/fc_lip0.8_w80.mat")
    # fc1 = io.loadmat("./models/fc_lip1.0_w80.mat")
    # fc5 = io.loadmat("./models/fc_lip5.0_w80.mat")

    fc02 = io.loadmat("./models/mnist_results_check/fc_lip0.2_w80.mat")
    fc03 = io.loadmat("./models/mnist_results_check/fc_lip0.3_w80.mat")
    fc04 = io.loadmat("./models/mnist_results_check/fc_lip0.4_w80.mat")
    fc05 = io.loadmat("./models/mnist_results_check/fc_lip0.5_w80.mat")
    fc08 = io.loadmat("./models/mnist_results_check/fc_lip0.8_w80.mat")
    fc1 = io.loadmat("./models/mnist_results_check/fc_lip1.0_w80.mat")
    fc5 = io.loadmat("./models/mnist_results_check/fc_lip5.0_w80.mat")
    fc10 = io.loadmat("./models/mnist_results_check/fc_lip10.0_w80.mat")
    fc50 = io.loadmat("./models/mnist_results_check/fc_lip50.0_w80.mat")


    lmt1 = io.loadmat("./models/mnist_results_check/lmt_c1_w80.mat")
    lmt10 = io.loadmat("./models/mnist_results_check/lmt_c10_w80.mat")
    lmt100 = io.loadmat("./models/mnist_results_check/lmt_c100_w80.mat")
    lmt1000 = io.loadmat("./models/mnist_results_check/lmt_c1000_w80.mat")

    
    plt.plot(fc02["epsilon"].T, fc02["errors"].T, label="0.2")
    # plt.plot(fc03["epsilon"].T, fc03["errors"].T)
    plt.plot(fc04["epsilon"].T, fc04["errors"].T, label="0.4")
    # plt.plot(fc05["epsilon"].T, fc05["errors"].T)
    plt.plot(fc08["epsilon"].T, fc08["errors"].T, label="0.8")
    # plt.plot(fc1["epsilon"].T, fc1["errors"].T)
    plt.plot(fc5["epsilon"].T, fc5["errors"].T, label="5")
    # plt.plot(fc10["epsilon"].T, fc10["errors"].T, label="10")
    # plt.plot(fc50["epsilon"].T, fc50["errors"].T, label="50")

    plt.plot(lmt1["epsilon"].T, lmt1["errors"].T, label="LMT1")
    plt.plot(lmt10["epsilon"].T, lmt10["errors"].T, label="LMT10")
    plt.plot(lmt100["epsilon"].T, lmt100["errors"].T, label="LMT1000")

    plt.legend()

    plt.show()