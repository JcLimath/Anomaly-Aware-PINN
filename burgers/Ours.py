"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch"""
import deepxde as dde
import numpy as np
import torch
import random


# seed = 123
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)


def gen_testdata():
    data = np.load("dataset/Burgers.npz")
    t, x, exact = data["t"], data["x"], data["usol"].T
    xx, tt = np.meshgrid(x, t)
    X = np.vstack((np.ravel(xx), np.ravel(tt))).T
    y = exact.flatten()[:, None]
    return X, y


def pde(x, y):
    dy_x = dde.grad.jacobian(y, x, i=0, j=0)
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    return dy_t + y * dy_x - 0.01 / np.pi * dy_xx


def sample_select_rate(iteration, T=300000):
    # 采样率算法,前两千次固定抛弃率
    if iteration > 1000:
        rate_tem = iteration / T + 0.7
        if rate_tem > 0.997:
            rate = 0.997
        else:
            rate = rate_tem
    else:
        rate = 0.7
    return rate


def get_weight(loss, iteration):
    # 使用一维方法
    res_number = 20000 + 40 + 80
    res_loss = loss[0:res_number]
    other_loss = loss[res_number:]

    loss_mean = np.mean(res_loss)
    loss_std = np.std(res_loss)
    loss_var = np.var(res_loss)
    loss_bound = loss_mean + 3 * loss_var

    vect_tem = res_loss - loss_bound * np.ones_like(res_loss)
    vect_tem[vect_tem > 0] = 1  # 判断是否全部都在阈值之内
    vect_tem[vect_tem < 0] = 0
    print("容忍度之外的点有", np.sum(vect_tem))

    # 获得采样率
    rate = sample_select_rate(iteration)
    select_number = round(rate * res_number)
    _, index = torch.topk(torch.Tensor(res_loss), select_number, dim=0, largest=False)  # 获得小点的坐标
    loss_weight = torch.zeros_like(torch.Tensor(res_loss))
    loss_weight[index] = torch.Tensor([1])

    return loss_weight


geom = dde.geometry.Interval(-1, 1)
timedomain = dde.geometry.TimeDomain(0, 0.99)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

bc = dde.icbc.DirichletBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)
ic = dde.icbc.IC(
    geomtime, lambda x: -np.sin(np.pi * x[:, 0:1]), lambda _, on_initial: on_initial
)

data = dde.data.TimePDE(
    geomtime, pde, [bc, ic], num_domain=20000, num_boundary=40, num_initial=80
)
net = dde.nn.FNN([2] + [200] * 4 + [1], "tanh", "Glorot normal")
model = dde.Model(data, net)

####### Train  ######
epochs = 300000
iteration = 500

model.compile("adam", lr=1e-4)
losshistory, train_state = model.train(iterations=1, model_restore_path=None,
                                       model_save_path=None)

for i in range(int(epochs / iteration)):

    # 计算权重
    x_tem = data.train_x
    f_tem = model.predict(x_tem, operator=pde)
    res_loss = np.square(f_tem)
    # res_loss = f_tem
    loss_weight = get_weight(res_loss, i * iteration)


    # 使用权重训练
    model.compile("adam", lr=1e-4, loss_weights=loss_weight)
    losshistory, train_state = model.train(iterations=iteration, model_restore_path=None,
                                           model_save_path=None)

np.savetxt(r"/hy-tmp/losshistory/ours_test_loss.txt", losshistory.loss_train)