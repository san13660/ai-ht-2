from functions import flatten_list_of_arrays, cost_function, back_propagation
from mnist_reader import load_mnist

import numpy as np
import time
import pickle
import math
import scipy
from scipy.optimize import minimize

NORMALIZATION_FACTOR = 1000.0
HIDDEN_NEURONS = 120
MAX_ITERATIONS = 1000

X_train, y_train = load_mnist('data/fashion', kind='train')

X = X_train / NORMALIZATION_FACTOR
m, n = X.shape
y_train = y_train.reshape(m, 1)
Y = (y_train == np.array(range(10))).astype(int)

layers = np.array([
    n,
    HIDDEN_NEURONS,
    10
])
theta_shapes = np.hstack((
    layers[1:].reshape(len(layers) - 1, 1),
    (layers[:-1] + 1).reshape(len(layers) - 1, 1)
))

flat_thetas = flatten_list_of_arrays([
    np.random.rand(*theta_shape)
    for theta_shape in theta_shapes
])

start = time.time()

print("Iniciando entrenamiento...")

result = minimize(
    fun = cost_function,
    x0 = flat_thetas,
    args = (theta_shapes, X, Y),
    method = 'L-BFGS-B',
    jac = back_propagation,
    options = {'disp': True, 'maxiter': MAX_ITERATIONS}
)

end = time.time()

print("\nTiempo en entrenamiento: {}s".format(end - start))

outfile = open('modelo', 'wb')
pickle.dump(result.x, outfile)
outfile.close()

print("Archivo del modelo creado!")