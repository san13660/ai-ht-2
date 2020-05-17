import numpy as np
import math
from functools import reduce
import scipy

sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))

flatten_list_of_arrays = lambda list_of_arrays: reduce(
    lambda acc, v: np.array([*acc.flatten(), *v.flatten()]),
    list_of_arrays
)

def inflate_matrixes(flat_thetas, shapes):
    layers = len(shapes) + 1
    sizes = [shape[0] * shape[1] for shape in shapes]
    steps = np.zeros(layers, dtype=int)

    for i in range(layers - 1):
        steps[i + 1] = steps[i] + sizes[i]

    return [
        flat_thetas[steps[i]: steps[i + 1]].reshape(*shapes[i])
        for i in range(layers - 1)
    ]

def feed_forward(thetas, X):
    a = [X]
    for i in range(len(thetas)):
        a.append(
            sigmoid(
                np.matmul(
                    np.hstack((np.ones(len(X)).reshape(len(X), 1), a[i])),
                    thetas[i].T
                )
            )
        )
    return a

def back_propagation(flat_thetas, shapes, X, Y):
    m, layers = len(X), len(shapes) + 1
    thetas = inflate_matrixes(flat_thetas, shapes)
    a = feed_forward(thetas, X)
    deltas = [*range(layers - 1), a[-1] - Y]

    for i in range(layers - 2, 0, -1):
        deltas[i] = (deltas[i + 1] @ np.delete(thetas[i], 0, 1)) * (a[i] * (1 - a[i]))
        
    new_deltas = []
    for n in range(layers - 1):
        new_deltas.append(
            (deltas[n + 1].T @ np.hstack((
                np.ones(len(a[n])).reshape(len(a[n]), 1),
                a[n]
            ))) / m
        )

    new_deltas = np.asarray(new_deltas)

    return flatten_list_of_arrays(
        new_deltas
    )

def cost_function(flat_thetas, shapes, X, Y):
    a = feed_forward(
        inflate_matrixes(flat_thetas, shapes),
        X
    )
    return -(Y * np.log(a[-1]) + (1 - Y) * np.log(1 - a[-1])).sum() / len(X)

def predict(flat_thetas, shapes, X):
    thetas = inflate_matrixes(flat_thetas, shapes)
    return feed_forward(thetas, X)[-1]