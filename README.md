# Linear Least Squares Fit

This python program gets arbitrarily generated linear data and uses linear least squares fit and the gradient descent approach to fit the data and decrease the error function.

## Comparing with gradient descent method:

After initializing the weights with chosen values, we found the fit for LMS. Now, when we train the data on gradient descent approach using batch learning, we find that after one epoch we get weights that are quite close to the optimal weights. We can stop here to get the linear fit, but if we keep training the algorithm using the updated weights, we get an even better fit. 

Hence, we can put a threshold value, below which the algorithm will stop. That is if the new weight is extremely close, we can stop the loop. If we keep running the algorithm using such threshold value, we get a fit very close to LMS one. The same can be observed in the plots at each stage.

### Refer:

1. [Gradient Descent Method](https://en.wikipedia.org/wiki/Gradient_descent)
2. [Linear Least Squares](https://en.wikipedia.org/wiki/Linear_least_squares)

### Libraries Used:

1. Numpy library is used to store and manipulate the data.

2. Matplotlib is used in order to plot the results.