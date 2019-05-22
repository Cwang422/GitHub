from numpy import exp, array, random, dot
training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
training_set_outputs = array([[0, 1, 1, 0]]).T
random.seed(1)
#synaptic_weights = 2 * random.random((3, 1)) - 1
#synaptic_weights= array([[0, 0, 0, 0]]).T
#print(synaptic_weights)
synaptic_weights = array([[0.00, 0.00, 0.00]]).T

for iteration in range(10000):
    output = 1 / (1 + exp(-(dot(training_set_inputs, synaptic_weights))))
    synaptic_weights += dot(training_set_inputs.T, (training_set_outputs - output) * output * (1 - output))
print (1 / (1 + exp(-(dot(array([1, 1, 1]), synaptic_weights)))))
print (synaptic_weights)