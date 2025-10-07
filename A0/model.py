import pandas as pd
import math
data = pd.read_excel('Boston_Housing.xlsx')
#print(data.shape)

y = data['MEDV'].to_numpy()
X = data.drop(columns=['MEDV'])

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Distribution of predictors and relationship with target(MEDV)
'''
for col in X.columns:
    fig, ax = plt.subplots(1, 2, figsize=(6,2))
    ax[0].hist(X[col])
    ax[1].scatter(X[col], y)
    fig.suptitle(col)
    plt.show()
'''
#   1.cost computation(MSE)
def compute_cost(X, y, w, b):
    m = X.shape[0]

    f_wb = np.dot(X, w) + b
    cost = np.sum(np.power(f_wb - y, 2))

    total_cost = 1 / (2 * m) * cost

    return total_cost

#   2.gradient computation
def compute_gradient(X, y, w, b):
    '''m, n = X.shape
    dj_dw = np.zeros((n,))
    dj_db = 0.

    err = (np.dot(X, w) + b) - y
    dj_dw = np.dot(X.T, err)    # dimension: (n,m)*(m,1)=(n,1)
    dj_db = np.sum(err)

    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_db, dj_dw'''

    m, n = X.shape
    err = np.dot(X, w) + b - y          # (m,)
    err = err.reshape(-1)               # 强制一维
    dj_dw = (1/m) * np.dot(X.T, err)    # (n,)
    dj_db = (1/m) * np.sum(err)
    return dj_db, dj_dw

#   3.gradient descent
def gradient_descent(X, y, w_in, b_in, alpha, num_iters):
    J_history = []
    w = w_in
    b = b_in

    for i in range(num_iters):
        dj_db, dj_dw = compute_gradient(X, y, w, b)

        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        cost = compute_cost(X, y, w, b)
        J_history.append(cost)

        if i % math.ceil(num_iters/10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}")

    return w, b, J_history

#   4.model training (iter, alpha, w_init, b_init), splitting of Training set and Test set
iterations = 10000
alpha = 1.0e-6
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
w_init = np.zeros(X_train.shape[1])
b_init = 0

#   4.1 data standardization
x_scaler = StandardScaler()
X_train_norm = x_scaler.fit_transform(X_train)
X_test_norm = x_scaler.transform(X_test)

'''y_scaler = StandardScaler()
y_train_norm = y_scaler.fit_transform(y_train.reshape(-1, 1))
y_test_norm = y_scaler.transform(y_test.reshape(-1, 1))'''

w_out, b_out, J_hist = gradient_descent(X_train, y_train, w_init, b_init, alpha, iterations)
print(f"Training result: w = {w_out}, b = {b_out}")
print(f"Training MSE = {J_hist[-1]}")

#   5.prediction
def predict(X, w, b):
    p = np.dot(X, w) + b
    return p

y_pred = predict(X_test, w_out, b_out)
print(y_pred)

#   6.result evaluation
def compute_mse(y1, y2):
    return np.mean(np.power((y1 - y2),2))
mse = compute_mse(y_test, y_pred)
print(mse)



