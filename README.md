# Rajesh

Some of the important formulas used in convolutional neural networks (CNN):

Output size of a convolutional layer: output_size = (input_size - filter_size + 2 * padding) / stride + 1

Number of parameters in a convolutional layer: number_of_params = (filter_size * input_channels + 1) * number_of_filters

Output size of a max-pooling layer: output_size = (input_size - pool_size) / stride + 1

Backpropagation for a convolutional layer:

4.1. Error term for the previous layer: delta_l = W_l^T * delta_l+1 * g'(z_l)

4.2. Gradient for the weights: dW_l = a_l-1 * delta_l

4.3. Gradient for the biases: db_l = sum(delta_l)

4.4. Error term for the current layer: delta_l = g'(z_l) * sum(W_l * delta_l+1)

Calculation of the loss function: loss = - (1 / N) * sum(y * log(y_hat) + (1 - y) * log(1 - y_hat))

where y is the true label, y_hat is the predicted label, and N is the number of samples.

Gradient descent with momentum: v = beta * v + (1 - beta) * dW, W = W - alpha * v

where v is the velocity, beta is the momentum parameter, dW is the gradient, W is the weight matrix, alpha is the learning rate.

Dropout regularization: W' = W * d
where d is a binary mask with a probability of keeping a neuron, p. The dropout probability, p, is usually set to 0.5.

-------------------------------------------------- 

There are several optimization techniques used in machine learning and deep learning to minimize the cost function and find the optimal weights for the model. Here are some of the most popular optimization techniques:

Gradient Descent:
Gradient descent is a first-order optimization algorithm that is widely used in machine learning and deep learning. It is an iterative algorithm that minimizes the cost function by adjusting the parameters in the direction of steepest descent. The update rule for gradient descent is as follows:

θ = θ − α * ∇J(θ)

where θ is the weight vector, α is the learning rate, and ∇J(θ) is the gradient of the cost function with respect to θ.

Stochastic Gradient Descent (SGD):
Stochastic gradient descent is a variant of gradient descent that randomly samples a subset of training data at each iteration. This makes the algorithm faster and more scalable, but it can also lead to noisy updates. The update rule for stochastic gradient descent is as follows:

θ = θ − α * ∇J(θ; xi, yi)

where θ is the weight vector, α is the learning rate, and ∇J(θ; xi, yi) is the gradient of the cost function with respect to θ, evaluated on a single training example (xi, yi).

Adam:
Adam (Adaptive Moment Estimation) is an optimization algorithm that computes adaptive learning rates for each parameter in the model. It combines the advantages of both gradient descent and stochastic gradient descent. The update rule for Adam is as follows:

m = β1 * m + (1 - β1) * ∇J(θ)
v = β2 * v + (1 - β2) * (∇J(θ))^2
θ = θ - α * m / (sqrt(v) + epsilon)

where θ is the weight vector, α is the learning rate, β1 and β2 are hyperparameters that control the decay rates of the two moving averages, m and v are the first and second moment estimates of the gradient, and epsilon is a small constant to prevent division by zero.

Adagrad:
Adagrad (Adaptive Gradient) is an optimization algorithm that adapts the learning rate for each parameter based on the historical gradients. It is particularly useful for sparse data, where some features may occur infrequently. The update rule for Adagrad is as follows:

G = G + (∇J(θ))^2
θ = θ - α * (∇J(θ) / sqrt(G + epsilon))

where θ is the weight vector, α is the learning rate, G is the sum of the squared gradients for each parameter, and epsilon is a small constant to prevent division by zero.

RMSprop:
RMSprop (Root Mean Square Propagation) is an optimization algorithm that adapts the learning rate for each parameter based on the historical gradients. It is similar to Adagrad, but it uses a moving average of the squared gradients instead of the sum of the squared gradients. The update rule for RMSprop is as follows:

G = β * G + (1 - β) * (∇J(θ))^2
θ = θ - α * (∇J(θ) / sqrt(G + epsilon))

where θ is the weight vector, α is the learning rate, β is a hyperparameter that controls the decay rate of the moving average, G is the moving average of the squared gradients, and epsilon is a small constant to prevent division by zero.

The main differences between these optimization techniques are in the way they update the weights and learning rates. Gradient descent




