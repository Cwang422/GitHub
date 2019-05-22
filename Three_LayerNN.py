from numpy import exp, array, random, dot
training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
training_set_outputs = array([[0, 1, 1, 0]]).T
# initialize the weights for training
random.seed(1)
# split 1 layer into 2 layers
# i,e matrix(3,5) * matrix (5,1) = matrix(3,1)
synaptic_weights_layer1 = 2 * random.random((3, 5)) - 1
synaptic_weights_layer2 = 2 * random.random((5, 1)) - 1


for iteration in range(10000):
    layer0 = training_set_inputs
    # using sigmoid function to train
    layer1 = 1 / (1 + exp(-(dot(layer0, synaptic_weights_layer1))))
    layer2 = 1 / (1 + exp(-(dot(layer1, synaptic_weights_layer2))))

    # using sigmoid function's derivative to update the new weights
    l2_error = training_set_outputs - layer2
    l2_delta = l2_error*layer2*(1-layer2)
    l1_error = l2_delta.dot(synaptic_weights_layer2.T)
    l1_delta = l1_error*layer1*(1-layer1)

    synaptic_weights_layer2 += layer1.T.dot(l2_delta)
    synaptic_weights_layer1 += layer0.T.dot(l1_delta)

# finally use the optimized weights to estimate [1,1,0]
print(1 / (1 + exp(-(dot(array([1, 0, 1]), dot(synaptic_weights_layer1,synaptic_weights_layer2))))))
