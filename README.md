# Rajesh

Some of the important formulas used in convolutional neural networks (CNN):

Output size of a convolutional layer: output_size = (input_size - filter_size + 2 * padding) / stride + 1
-------------------------------------------------------------------------------------------------------------------------------------
Number of parameters in a convolutional layer: number_of_params = (filter_size * input_channels + 1) * number_of_filters

Output size of a max-pooling layer: output_size = (input_size - pool_size) / stride + 1
**-------------------------------------------------------------------------------------------------------------------------------------**

Backpropagation for a convolutional layer:

4.1. Error term for the previous layer: delta_l = W_l^T * delta_l+1 * g'(z_l)

4.2. Gradient for the weights: dW_l = a_l-1 * delta_l

4.3. Gradient for the biases: db_l = sum(delta_l)

4.4. Error term for the current layer: delta_l = g'(z_l) * sum(W_l * delta_l+1)

**-------------------------------------------------------------------------------------------------------------------------------------**

Calculation of the loss function: loss = - (1 / N) * sum(y * log(y_hat) + (1 - y) * log(1 - y_hat))

where y is the true label, y_hat is the predicted label, and N is the number of samples.
**-------------------------------------------------------------------------------------------------------------------------------------**


Gradient descent with momentum: v = beta * v + (1 - beta) * dW, W = W - alpha * v

where v is the velocity, beta is the momentum parameter, dW is the gradient, W is the weight matrix, alpha is the learning rate.
**-------------------------------------------------------------------------------------------------------------------------------------**

Dropout regularization: W' = W * d
where d is a binary mask with a probability of keeping a neuron, p. The dropout probability, p, is usually set to 0.5.
