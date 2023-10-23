

import numpy as np
from freeplot.base import FreePlot
from freeplot.utils import import_pickle



def relative_error(x, y):
    x = x[y > 1e-6]
    y = y[y > 1e-6]
    return np.mean(np.abs((x - y)) / np.abs(y))

def mean_square_error(x, y):
    return np.mean((x - y) ** 2)

def mean_abs_error(x, y):
    return np.mean(np.abs(x - y))



# solution curves

wrappers = ['identity', 'refine', 'tanh']
epss = ['1e-3', '1e-6', '1e-9']

legends = {
    'exact': 'Exact solution',
    'identity': r"$r^2$",
    'log': r"$\log (r^2 + 1)$",
    'refine': r"$-\frac{1}{r^2 + 1} + 1$",
    'tanh': r"$\tanh(r^2)$"
}

titles=(r"$\epsilon = 10^{-3}$", r"$\epsilon = 10^{-6}$", r"$\epsilon = 10^{-9}$", r"$\epsilon = 10^{-3}$", r"$\epsilon = 10^{-6}$", r"$\epsilon = 10^{-9}$")
fp = FreePlot(shape=(3, 3), figsize=(8, 6.3), latex=True)
for i, eps in enumerate(epss):
    flag = True
    for j, wrapper in enumerate(wrappers):
        data = import_pickle(f"./logs/{wrapper}-{eps}/preds.equ")
        pred, target = data['pred'], data['real']

        if flag:
            fp.lineplot(target['x'], target['y'], index=(0, i), marker='', label=legends['exact'])
            flag = False
        fp.lineplot(pred['x'], pred['y'], index=(0, i), marker='', label=legends[wrapper])

        x, y = pred['y'], target['y']
        rel_error = relative_error(x, y)
        mse = mean_square_error(x, y)
        mae = mean_abs_error(x, y)
        print(f"{eps}-{wrapper} >>> rel_error: {rel_error}, mse: {mse}, mae: {mae}")
        fp.set_label('Computational domain', index=(0, i), axis='x')
        fp.set(index=(0, i), title=titles[i])


for i, eps in enumerate(epss):
    for wrapper, color in zip(wrappers, fp.colors[1:]):
        data = import_pickle(f"./logs/{wrapper}-{eps}/preds.equ")
        pred, target = data['pred'], data['real']

        y = np.abs(pred['y'] -  target['y'])
        fp.lineplot(pred['x'], y, index=(1, i), marker='', label=legends[wrapper], color=color)
        fp.set_label('Computational domain', index=(1, i), axis='x')

for i, eps in enumerate(epss):
    flag = True
    for wrapper, colors in zip(wrappers, fp.colors[1:]):
        data = import_pickle(f"./logs/{wrapper}-{eps}/loss.equ")
        T, loss = data['T'], data['Loss']
        T = np.array(T) * 50

        fp.lineplot(T, loss, index=(2, i), marker='', label=legends[wrapper], color=colors)

    fp.set_label('Iterations', index=(2, i), axis='x')
    fp.ticklabel_format(index=(2, i), axis='x')
    fp.set_lim((-0.05, 0.5), index=(2, i))


fp.set_label('Solution', index=(0, 0), axis='y')
fp.set_label('MAE', index=(1, 0), axis='y')
fp.set_label('Training loss', index=(2, 0), axis='y')
fp.legend(0.32, 0.93, 4)
fp.subplots_adjust(hspace=.3)
# fp.set_title()
# fp.show(tight_layout=False)
fp.savefig("./test.pdf", tight_layout=False)
