#!/usr/bin/python
import fann

connection_rate = 1
learning_rate = 0.7
num_input = 2
num_neurons_hidden = 4
num_output = 1

desired_error = 0.0001
max_iterations = 100000
iterations_between_reports = 1000

ann = fann.create(connection_rate, learning_rate, (num_input, num_neurons_hidden, num_output))

ann.train_on_file("datasets/xor.data", max_iterations, iterations_between_reports, desired_error)

ann.save("xor_float.net")

ann.destroy()
