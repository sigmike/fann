#!/usr/bin/python
import fann

connection_rate = 1
learning_rate = 0.7
num_layers = 3
num_input = 2
num_neurons_hidden = 4
num_output = 1


ann = fann.fann_create(connection_rate, learning_rate, num_layers,num_input, num_neurons_hidden, num_output)

desired_error = 0.0001
max_iterations = 500000
iterations_between_reports = 1000
fann.fann_train_on_file(ann, "../examples/xor.data", max_iterations, iterations_between_reports, desired_error)
fann.fann_save(ann, "xor_float.net")

fann.fann_destroy(ann)
