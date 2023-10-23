


# %%

import numpy as np
from freeplot.base import FreePlot
from freeplot.utils import import_pickle


# %%

def relative_error(x, y):
    x = x[y > 1e-6]
    y = y[y > 1e-6]
    return np.mean(np.abs((x - y)) / np.abs(y))

def mean_square_error(x, y):
    return np.mean((x - y) ** 2)

def mean_abs_error(x, y):
    return np.mean(np.abs(x - y))

# %%

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


fp = FreePlot(shape=(1, 3), figsize=(6.5, 2), titles=(r"$10^{-3}$", r"$10^{-6}$", r"$10^{-9}$"), latex=True)
for i, eps in enumerate(epss):
    flag = True
    for j, wrapper in enumerate(wrappers):
        data = import_pickle(f"./logs/{wrapper}-{eps}/preds.equ")
        pred, target = data['pred'], data['real']

        if flag:
            fp.lineplot(target['x'], target['y'], index=(0, i), marker='', label=legends['exact'])
            flag = False
        fp.lineplot(pred['x'], pred['y'], index=(0, i), linestyle=':', marker='', label=legends[wrapper])

        x, y = pred['y'], target['y']
        rel_error = relative_error(x, y)
        mse = mean_square_error(x, y)
        mae = mean_abs_error(x, y)
        print(f"{eps}-{wrapper} >>> rel_error: {rel_error}, mse: {mse}, mae: {mae}")
        fp.set_label('Computational domain', index=(0, i), axis='x')

fp.set_label('Solution', axis='y')
fp[0, 0].legend()
fp.set_title()
# fp.show()
fp.savefig("./test.pdf")

# %%

# loss curves


wrappers = ['identity', 'refine', 'tanh']
epss = ['1e-3', '1e-6', '1e-9']

legends = {
    'exact': 'Exact solution',
    'identity': r"$r^2$",
    'log': r"$\log (r^2 + 1)$",
    'refine': r"$-\frac{1}{r^2 + 1} + 1$",
    'tanh': r"$\tanh(r^2)$"
}

for eps in epss:
    fp = FreePlot(dpi=300)
    flag = True
    for wrapper in wrappers:
        data = import_pickle(f"./logs/{wrapper}-{eps}/loss.equ")
        T, loss = data['T'], data['Loss']
        T = np.array(T) * 50

        fp.lineplot(T, loss, marker='', label=legends[wrapper])

    fp.set_label('Iterations', axis='x')
    fp.set_label('Training loss', axis='y')
    fp[0, 0].legend()
    fp.show()
