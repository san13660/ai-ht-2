from functions import predict
from mnist_reader import load_mnist

from scipy.optimize import minimize
import numpy as np
import pickle

NORMALIZATION_FACTOR = 1000.0
HIDDEN_NEURONS = 125

X_test, y_test = load_mnist('data/fashion', kind='t10k')
X = X_test / NORMALIZATION_FACTOR
m, n = X.shape

layers = np.array([
    n,
    HIDDEN_NEURONS,
    10
])
theta_shapes = np.hstack((
    layers[1:].reshape(len(layers) - 1, 1),
    (layers[:-1] + 1).reshape(len(layers) - 1, 1)
))

openfile = open("trained_model", "rb")
thetas_trained = pickle.load(openfile)

predictions = predict(thetas_trained, theta_shapes, X)

predictions_cleaned = np.array([])

for prediction in predictions:
    condition = (prediction == np.amax(prediction))
    index = np.where(condition)[0]
    predictions_cleaned = np.concatenate((predictions_cleaned,index))

matches = np.sum(predictions_cleaned == y_test)
precision = matches/len(predictions)

print('Predicciones acertadas: {:d}/{:d}'.format(matches, len(predictions)))
print('Porcentaje de precision: {:.2f}%'.format(precision*100))