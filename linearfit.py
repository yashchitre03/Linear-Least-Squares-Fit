import numpy as np
import matplotlib.pyplot as plt

np.random.seed(100)
X = np.arange(1, 51, 1).reshape(50, 1)
u = np.random.uniform(-1, 1, (50, 1))
Y = u + X

X_temp = np.vstack((np.ones((1, 50)), X.T))
X_psuedo_inverse = np.matmul(X_temp.T, np.linalg.inv(np.matmul(X_temp, X_temp.T)))

W = np.matmul(Y.reshape(1, 50), X_psuedo_inverse)

f_X = W[0][1]*X[:, 0] + W[0][0]
plt.plot(X[:, 0], Y[:, 0], 'o')
plt.plot(X[:, 0], f_X, 'r', label = 'Line to fit data')
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Linear Least Squares Fit")
plt.legend()
plt.show()


w = np.array([0.15, 0.6])
eta = 0.00001

f_X = w[1]*X[:, 0] + w[0]
plt.plot(X[:, 0], Y[:, 0], 'o')
plt.plot(X[:, 0], f_X, 'r', label = 'Line to fit data')
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Gradient Descent Fit before training")
plt.legend()
plt.show()
print(w, eta)

z = 0
while True:
    z+=1
    grad = np.zeros((2,))
    for i in range(50):
        grad[0] += (Y[i, 0] - w[0] - w[1]*X[i, 0])
    grad[0] *= -2
    
    for i in range(50):
        grad[1] += (Y[i, 0] - w[0] - w[1]*X[i, 0])*X[i, 0]
    grad[1] *= -2
    
    delta_w = eta * grad
    new_w = w - delta_w
    
    if np.linalg.norm(w - new_w) < 0.0001:
        break
    else:
        w = new_w
        wtemp = new_w
    if z == 1:
        f_X = w[1]*X[:, 0] + w[0]
        plt.plot(X[:, 0], Y[:, 0], 'o')
        plt.plot(X[:, 0], f_X, 'r', label = 'Line to fit data')
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.title("Gradient Descent Fit after 1 epoch")
        plt.legend()
        plt.show()
        
f_X = w[1]*X[:, 0] + w[0]
plt.plot(X[:, 0], Y[:, 0], 'o')
plt.plot(X[:, 0], f_X, 'r', label = 'Line to fit data')
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title(f"Gradient Descent Fit after {z} epochs")
plt.legend()
plt.show()
        
print("DIFF", W - w.reshape(1, 2))
     



