# This file was created automatically by SWIG.
# Don't modify this file, modify the SWIG interface instead.
# This file is compatible with both classic and new-style classes.

import _libfann

def _swig_setattr(self,class_type,name,value):
    if (name == "this"):
        if isinstance(value, class_type):
            self.__dict__[name] = value.this
            if hasattr(value,"thisown"): self.__dict__["thisown"] = value.thisown
            del value.thisown
            return
    method = class_type.__swig_setmethods__.get(name,None)
    if method: return method(self,value)
    self.__dict__[name] = value

def _swig_getattr(self,class_type,name):
    method = class_type.__swig_getmethods__.get(name,None)
    if method: return method(self)
    raise AttributeError,name

import types
try:
    _object = types.ObjectType
    _newclass = 1
except AttributeError:
    class _object : pass
    _newclass = 0
del types


NULL = _libfann.NULL

fann_create_array = _libfann.fann_create_array

fann_create_shortcut_array = _libfann.fann_create_shortcut_array

fann_run_old = _libfann.fann_run_old

fann_destroy = _libfann.fann_destroy

fann_randomize_weights = _libfann.fann_randomize_weights

fann_init_weights = _libfann.fann_init_weights

fann_print_connections = _libfann.fann_print_connections

fann_create_from_file = _libfann.fann_create_from_file

fann_save = _libfann.fann_save

fann_save_to_fixed = _libfann.fann_save_to_fixed

fann_train = _libfann.fann_train

fann_test_old = _libfann.fann_test_old

fann_get_error = _libfann.fann_get_error

fann_get_MSE = _libfann.fann_get_MSE

fann_reset_error = _libfann.fann_reset_error

fann_reset_MSE = _libfann.fann_reset_MSE

fann_read_train_from_file = _libfann.fann_read_train_from_file

fann_destroy_train = _libfann.fann_destroy_train

fann_train_epoch = _libfann.fann_train_epoch

fann_test_data = _libfann.fann_test_data

fann_train_on_data = _libfann.fann_train_on_data

fann_train_on_data_callback = _libfann.fann_train_on_data_callback

fann_train_on_file = _libfann.fann_train_on_file

fann_train_on_file_callback = _libfann.fann_train_on_file_callback

fann_shuffle_train_data = _libfann.fann_shuffle_train_data

fann_merge_train_data = _libfann.fann_merge_train_data

fann_duplicate_train_data = _libfann.fann_duplicate_train_data

fann_save_train = _libfann.fann_save_train

fann_save_train_to_fixed = _libfann.fann_save_train_to_fixed

fann_cascadetrain_on_data_callback = _libfann.fann_cascadetrain_on_data_callback

fann_print_parameters = _libfann.fann_print_parameters

fann_get_training_algorithm = _libfann.fann_get_training_algorithm

fann_set_training_algorithm = _libfann.fann_set_training_algorithm

fann_get_learning_rate = _libfann.fann_get_learning_rate

fann_set_learning_rate = _libfann.fann_set_learning_rate

fann_get_activation_function_hidden = _libfann.fann_get_activation_function_hidden

fann_set_activation_function_hidden = _libfann.fann_set_activation_function_hidden

fann_get_activation_function_output = _libfann.fann_get_activation_function_output

fann_set_activation_function_output = _libfann.fann_set_activation_function_output

fann_get_activation_steepness_hidden = _libfann.fann_get_activation_steepness_hidden

fann_set_activation_steepness_hidden = _libfann.fann_set_activation_steepness_hidden

fann_get_activation_steepness_output = _libfann.fann_get_activation_steepness_output

fann_set_activation_steepness_output = _libfann.fann_set_activation_steepness_output

fann_get_activation_hidden_steepness = _libfann.fann_get_activation_hidden_steepness

fann_set_activation_hidden_steepness = _libfann.fann_set_activation_hidden_steepness

fann_get_activation_output_steepness = _libfann.fann_get_activation_output_steepness

fann_set_activation_output_steepness = _libfann.fann_set_activation_output_steepness

fann_set_train_error_function = _libfann.fann_set_train_error_function

fann_get_train_error_function = _libfann.fann_get_train_error_function

fann_get_quickprop_decay = _libfann.fann_get_quickprop_decay

fann_set_quickprop_decay = _libfann.fann_set_quickprop_decay

fann_get_quickprop_mu = _libfann.fann_get_quickprop_mu

fann_set_quickprop_mu = _libfann.fann_set_quickprop_mu

fann_get_rprop_increase_factor = _libfann.fann_get_rprop_increase_factor

fann_set_rprop_increase_factor = _libfann.fann_set_rprop_increase_factor

fann_get_rprop_decrease_factor = _libfann.fann_get_rprop_decrease_factor

fann_set_rprop_decrease_factor = _libfann.fann_set_rprop_decrease_factor

fann_get_rprop_delta_min = _libfann.fann_get_rprop_delta_min

fann_set_rprop_delta_min = _libfann.fann_set_rprop_delta_min

fann_get_rprop_delta_max = _libfann.fann_get_rprop_delta_max

fann_set_rprop_delta_max = _libfann.fann_set_rprop_delta_max

fann_get_num_input = _libfann.fann_get_num_input

fann_get_num_output = _libfann.fann_get_num_output

fann_get_total_neurons = _libfann.fann_get_total_neurons

fann_get_total_connections = _libfann.fann_get_total_connections

fann_set_error_log = _libfann.fann_set_error_log

fann_get_errno = _libfann.fann_get_errno

fann_reset_errno = _libfann.fann_reset_errno

fann_reset_errstr = _libfann.fann_reset_errstr

fann_get_errstr = _libfann.fann_get_errstr

fann_print_error = _libfann.fann_print_error
class fann_neuron(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, fann_neuron, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, fann_neuron, name)
    def __repr__(self):
        return "<C fann_neuron instance at %s>" % (self.this,)
    __swig_setmethods__["first_con"] = _libfann.fann_neuron_first_con_set
    __swig_getmethods__["first_con"] = _libfann.fann_neuron_first_con_get
    if _newclass:first_con = property(_libfann.fann_neuron_first_con_get, _libfann.fann_neuron_first_con_set)
    __swig_setmethods__["last_con"] = _libfann.fann_neuron_last_con_set
    __swig_getmethods__["last_con"] = _libfann.fann_neuron_last_con_get
    if _newclass:last_con = property(_libfann.fann_neuron_last_con_get, _libfann.fann_neuron_last_con_set)
    __swig_setmethods__["sum"] = _libfann.fann_neuron_sum_set
    __swig_getmethods__["sum"] = _libfann.fann_neuron_sum_get
    if _newclass:sum = property(_libfann.fann_neuron_sum_get, _libfann.fann_neuron_sum_set)
    __swig_setmethods__["value"] = _libfann.fann_neuron_value_set
    __swig_getmethods__["value"] = _libfann.fann_neuron_value_get
    if _newclass:value = property(_libfann.fann_neuron_value_get, _libfann.fann_neuron_value_set)
    def __init__(self, *args):
        _swig_setattr(self, fann_neuron, 'this', _libfann.new_fann_neuron(*args))
        _swig_setattr(self, fann_neuron, 'thisown', 1)
    def __del__(self, destroy=_libfann.delete_fann_neuron):
        try:
            if self.thisown: destroy(self)
        except: pass

class fann_neuronPtr(fann_neuron):
    def __init__(self, this):
        _swig_setattr(self, fann_neuron, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, fann_neuron, 'thisown', 0)
        _swig_setattr(self, fann_neuron,self.__class__,fann_neuron)
_libfann.fann_neuron_swigregister(fann_neuronPtr)

class fann_layer(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, fann_layer, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, fann_layer, name)
    def __repr__(self):
        return "<C fann_layer instance at %s>" % (self.this,)
    __swig_setmethods__["first_neuron"] = _libfann.fann_layer_first_neuron_set
    __swig_getmethods__["first_neuron"] = _libfann.fann_layer_first_neuron_get
    if _newclass:first_neuron = property(_libfann.fann_layer_first_neuron_get, _libfann.fann_layer_first_neuron_set)
    __swig_setmethods__["last_neuron"] = _libfann.fann_layer_last_neuron_set
    __swig_getmethods__["last_neuron"] = _libfann.fann_layer_last_neuron_get
    if _newclass:last_neuron = property(_libfann.fann_layer_last_neuron_get, _libfann.fann_layer_last_neuron_set)
    def __init__(self, *args):
        _swig_setattr(self, fann_layer, 'this', _libfann.new_fann_layer(*args))
        _swig_setattr(self, fann_layer, 'thisown', 1)
    def __del__(self, destroy=_libfann.delete_fann_layer):
        try:
            if self.thisown: destroy(self)
        except: pass

class fann_layerPtr(fann_layer):
    def __init__(self, this):
        _swig_setattr(self, fann_layer, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, fann_layer, 'thisown', 0)
        _swig_setattr(self, fann_layer,self.__class__,fann_layer)
_libfann.fann_layer_swigregister(fann_layerPtr)

class fann(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, fann, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, fann, name)
    def __repr__(self):
        return "<C fann instance at %s>" % (self.this,)
    __swig_setmethods__["errno_f"] = _libfann.fann_errno_f_set
    __swig_getmethods__["errno_f"] = _libfann.fann_errno_f_get
    if _newclass:errno_f = property(_libfann.fann_errno_f_get, _libfann.fann_errno_f_set)
    __swig_setmethods__["error_log"] = _libfann.fann_error_log_set
    __swig_getmethods__["error_log"] = _libfann.fann_error_log_get
    if _newclass:error_log = property(_libfann.fann_error_log_get, _libfann.fann_error_log_set)
    __swig_setmethods__["errstr"] = _libfann.fann_errstr_set
    __swig_getmethods__["errstr"] = _libfann.fann_errstr_get
    if _newclass:errstr = property(_libfann.fann_errstr_get, _libfann.fann_errstr_set)
    __swig_setmethods__["learning_rate"] = _libfann.fann_learning_rate_set
    __swig_getmethods__["learning_rate"] = _libfann.fann_learning_rate_get
    if _newclass:learning_rate = property(_libfann.fann_learning_rate_get, _libfann.fann_learning_rate_set)
    __swig_setmethods__["connection_rate"] = _libfann.fann_connection_rate_set
    __swig_getmethods__["connection_rate"] = _libfann.fann_connection_rate_get
    if _newclass:connection_rate = property(_libfann.fann_connection_rate_get, _libfann.fann_connection_rate_set)
    __swig_setmethods__["shortcut_connections"] = _libfann.fann_shortcut_connections_set
    __swig_getmethods__["shortcut_connections"] = _libfann.fann_shortcut_connections_get
    if _newclass:shortcut_connections = property(_libfann.fann_shortcut_connections_get, _libfann.fann_shortcut_connections_set)
    __swig_setmethods__["first_layer"] = _libfann.fann_first_layer_set
    __swig_getmethods__["first_layer"] = _libfann.fann_first_layer_get
    if _newclass:first_layer = property(_libfann.fann_first_layer_get, _libfann.fann_first_layer_set)
    __swig_setmethods__["last_layer"] = _libfann.fann_last_layer_set
    __swig_getmethods__["last_layer"] = _libfann.fann_last_layer_get
    if _newclass:last_layer = property(_libfann.fann_last_layer_get, _libfann.fann_last_layer_set)
    __swig_setmethods__["total_neurons"] = _libfann.fann_total_neurons_set
    __swig_getmethods__["total_neurons"] = _libfann.fann_total_neurons_get
    if _newclass:total_neurons = property(_libfann.fann_total_neurons_get, _libfann.fann_total_neurons_set)
    __swig_setmethods__["num_input"] = _libfann.fann_num_input_set
    __swig_getmethods__["num_input"] = _libfann.fann_num_input_get
    if _newclass:num_input = property(_libfann.fann_num_input_get, _libfann.fann_num_input_set)
    __swig_setmethods__["num_output"] = _libfann.fann_num_output_set
    __swig_getmethods__["num_output"] = _libfann.fann_num_output_get
    if _newclass:num_output = property(_libfann.fann_num_output_get, _libfann.fann_num_output_set)
    __swig_setmethods__["weights"] = _libfann.fann_weights_set
    __swig_getmethods__["weights"] = _libfann.fann_weights_get
    if _newclass:weights = property(_libfann.fann_weights_get, _libfann.fann_weights_set)
    __swig_setmethods__["connections"] = _libfann.fann_connections_set
    __swig_getmethods__["connections"] = _libfann.fann_connections_get
    if _newclass:connections = property(_libfann.fann_connections_get, _libfann.fann_connections_set)
    __swig_setmethods__["train_errors"] = _libfann.fann_train_errors_set
    __swig_getmethods__["train_errors"] = _libfann.fann_train_errors_get
    if _newclass:train_errors = property(_libfann.fann_train_errors_get, _libfann.fann_train_errors_set)
    __swig_setmethods__["activation_function_hidden"] = _libfann.fann_activation_function_hidden_set
    __swig_getmethods__["activation_function_hidden"] = _libfann.fann_activation_function_hidden_get
    if _newclass:activation_function_hidden = property(_libfann.fann_activation_function_hidden_get, _libfann.fann_activation_function_hidden_set)
    __swig_setmethods__["activation_function_output"] = _libfann.fann_activation_function_output_set
    __swig_getmethods__["activation_function_output"] = _libfann.fann_activation_function_output_get
    if _newclass:activation_function_output = property(_libfann.fann_activation_function_output_get, _libfann.fann_activation_function_output_set)
    __swig_setmethods__["activation_steepness_hidden"] = _libfann.fann_activation_steepness_hidden_set
    __swig_getmethods__["activation_steepness_hidden"] = _libfann.fann_activation_steepness_hidden_get
    if _newclass:activation_steepness_hidden = property(_libfann.fann_activation_steepness_hidden_get, _libfann.fann_activation_steepness_hidden_set)
    __swig_setmethods__["activation_steepness_output"] = _libfann.fann_activation_steepness_output_set
    __swig_getmethods__["activation_steepness_output"] = _libfann.fann_activation_steepness_output_get
    if _newclass:activation_steepness_output = property(_libfann.fann_activation_steepness_output_get, _libfann.fann_activation_steepness_output_set)
    __swig_setmethods__["training_algorithm"] = _libfann.fann_training_algorithm_set
    __swig_getmethods__["training_algorithm"] = _libfann.fann_training_algorithm_get
    if _newclass:training_algorithm = property(_libfann.fann_training_algorithm_get, _libfann.fann_training_algorithm_set)
    __swig_setmethods__["activation_results_hidden"] = _libfann.fann_activation_results_hidden_set
    __swig_getmethods__["activation_results_hidden"] = _libfann.fann_activation_results_hidden_get
    if _newclass:activation_results_hidden = property(_libfann.fann_activation_results_hidden_get, _libfann.fann_activation_results_hidden_set)
    __swig_setmethods__["activation_values_hidden"] = _libfann.fann_activation_values_hidden_set
    __swig_getmethods__["activation_values_hidden"] = _libfann.fann_activation_values_hidden_get
    if _newclass:activation_values_hidden = property(_libfann.fann_activation_values_hidden_get, _libfann.fann_activation_values_hidden_set)
    __swig_setmethods__["activation_results_output"] = _libfann.fann_activation_results_output_set
    __swig_getmethods__["activation_results_output"] = _libfann.fann_activation_results_output_get
    if _newclass:activation_results_output = property(_libfann.fann_activation_results_output_get, _libfann.fann_activation_results_output_set)
    __swig_setmethods__["activation_values_output"] = _libfann.fann_activation_values_output_set
    __swig_getmethods__["activation_values_output"] = _libfann.fann_activation_values_output_get
    if _newclass:activation_values_output = property(_libfann.fann_activation_values_output_get, _libfann.fann_activation_values_output_set)
    __swig_setmethods__["total_connections"] = _libfann.fann_total_connections_set
    __swig_getmethods__["total_connections"] = _libfann.fann_total_connections_get
    if _newclass:total_connections = property(_libfann.fann_total_connections_get, _libfann.fann_total_connections_set)
    __swig_setmethods__["output"] = _libfann.fann_output_set
    __swig_getmethods__["output"] = _libfann.fann_output_get
    if _newclass:output = property(_libfann.fann_output_get, _libfann.fann_output_set)
    __swig_setmethods__["num_MSE"] = _libfann.fann_num_MSE_set
    __swig_getmethods__["num_MSE"] = _libfann.fann_num_MSE_get
    if _newclass:num_MSE = property(_libfann.fann_num_MSE_get, _libfann.fann_num_MSE_set)
    __swig_setmethods__["MSE_value"] = _libfann.fann_MSE_value_set
    __swig_getmethods__["MSE_value"] = _libfann.fann_MSE_value_get
    if _newclass:MSE_value = property(_libfann.fann_MSE_value_get, _libfann.fann_MSE_value_set)
    __swig_setmethods__["num_bit_fail"] = _libfann.fann_num_bit_fail_set
    __swig_getmethods__["num_bit_fail"] = _libfann.fann_num_bit_fail_get
    if _newclass:num_bit_fail = property(_libfann.fann_num_bit_fail_get, _libfann.fann_num_bit_fail_set)
    __swig_setmethods__["train_error_function"] = _libfann.fann_train_error_function_set
    __swig_getmethods__["train_error_function"] = _libfann.fann_train_error_function_get
    if _newclass:train_error_function = property(_libfann.fann_train_error_function_get, _libfann.fann_train_error_function_set)
    __swig_setmethods__["cascade_change_fraction"] = _libfann.fann_cascade_change_fraction_set
    __swig_getmethods__["cascade_change_fraction"] = _libfann.fann_cascade_change_fraction_get
    if _newclass:cascade_change_fraction = property(_libfann.fann_cascade_change_fraction_get, _libfann.fann_cascade_change_fraction_set)
    __swig_setmethods__["cascade_stagnation_epochs"] = _libfann.fann_cascade_stagnation_epochs_set
    __swig_getmethods__["cascade_stagnation_epochs"] = _libfann.fann_cascade_stagnation_epochs_get
    if _newclass:cascade_stagnation_epochs = property(_libfann.fann_cascade_stagnation_epochs_get, _libfann.fann_cascade_stagnation_epochs_set)
    __swig_setmethods__["cascade_num_candidates"] = _libfann.fann_cascade_num_candidates_set
    __swig_getmethods__["cascade_num_candidates"] = _libfann.fann_cascade_num_candidates_get
    if _newclass:cascade_num_candidates = property(_libfann.fann_cascade_num_candidates_get, _libfann.fann_cascade_num_candidates_set)
    __swig_setmethods__["cascade_best_candidate"] = _libfann.fann_cascade_best_candidate_set
    __swig_getmethods__["cascade_best_candidate"] = _libfann.fann_cascade_best_candidate_get
    if _newclass:cascade_best_candidate = property(_libfann.fann_cascade_best_candidate_get, _libfann.fann_cascade_best_candidate_set)
    __swig_setmethods__["cascade_candidate_limit"] = _libfann.fann_cascade_candidate_limit_set
    __swig_getmethods__["cascade_candidate_limit"] = _libfann.fann_cascade_candidate_limit_get
    if _newclass:cascade_candidate_limit = property(_libfann.fann_cascade_candidate_limit_get, _libfann.fann_cascade_candidate_limit_set)
    __swig_setmethods__["cascade_weight_multiplier"] = _libfann.fann_cascade_weight_multiplier_set
    __swig_getmethods__["cascade_weight_multiplier"] = _libfann.fann_cascade_weight_multiplier_get
    if _newclass:cascade_weight_multiplier = property(_libfann.fann_cascade_weight_multiplier_get, _libfann.fann_cascade_weight_multiplier_set)
    __swig_setmethods__["cascade_candidate_scores"] = _libfann.fann_cascade_candidate_scores_set
    __swig_getmethods__["cascade_candidate_scores"] = _libfann.fann_cascade_candidate_scores_get
    if _newclass:cascade_candidate_scores = property(_libfann.fann_cascade_candidate_scores_get, _libfann.fann_cascade_candidate_scores_set)
    __swig_setmethods__["total_neurons_allocated"] = _libfann.fann_total_neurons_allocated_set
    __swig_getmethods__["total_neurons_allocated"] = _libfann.fann_total_neurons_allocated_get
    if _newclass:total_neurons_allocated = property(_libfann.fann_total_neurons_allocated_get, _libfann.fann_total_neurons_allocated_set)
    __swig_setmethods__["total_connections_allocated"] = _libfann.fann_total_connections_allocated_set
    __swig_getmethods__["total_connections_allocated"] = _libfann.fann_total_connections_allocated_get
    if _newclass:total_connections_allocated = property(_libfann.fann_total_connections_allocated_get, _libfann.fann_total_connections_allocated_set)
    __swig_setmethods__["quickprop_decay"] = _libfann.fann_quickprop_decay_set
    __swig_getmethods__["quickprop_decay"] = _libfann.fann_quickprop_decay_get
    if _newclass:quickprop_decay = property(_libfann.fann_quickprop_decay_get, _libfann.fann_quickprop_decay_set)
    __swig_setmethods__["quickprop_mu"] = _libfann.fann_quickprop_mu_set
    __swig_getmethods__["quickprop_mu"] = _libfann.fann_quickprop_mu_get
    if _newclass:quickprop_mu = property(_libfann.fann_quickprop_mu_get, _libfann.fann_quickprop_mu_set)
    __swig_setmethods__["rprop_increase_factor"] = _libfann.fann_rprop_increase_factor_set
    __swig_getmethods__["rprop_increase_factor"] = _libfann.fann_rprop_increase_factor_get
    if _newclass:rprop_increase_factor = property(_libfann.fann_rprop_increase_factor_get, _libfann.fann_rprop_increase_factor_set)
    __swig_setmethods__["rprop_decrease_factor"] = _libfann.fann_rprop_decrease_factor_set
    __swig_getmethods__["rprop_decrease_factor"] = _libfann.fann_rprop_decrease_factor_get
    if _newclass:rprop_decrease_factor = property(_libfann.fann_rprop_decrease_factor_get, _libfann.fann_rprop_decrease_factor_set)
    __swig_setmethods__["rprop_delta_min"] = _libfann.fann_rprop_delta_min_set
    __swig_getmethods__["rprop_delta_min"] = _libfann.fann_rprop_delta_min_get
    if _newclass:rprop_delta_min = property(_libfann.fann_rprop_delta_min_get, _libfann.fann_rprop_delta_min_set)
    __swig_setmethods__["rprop_delta_max"] = _libfann.fann_rprop_delta_max_set
    __swig_getmethods__["rprop_delta_max"] = _libfann.fann_rprop_delta_max_get
    if _newclass:rprop_delta_max = property(_libfann.fann_rprop_delta_max_get, _libfann.fann_rprop_delta_max_set)
    __swig_setmethods__["rprop_delta_zero"] = _libfann.fann_rprop_delta_zero_set
    __swig_getmethods__["rprop_delta_zero"] = _libfann.fann_rprop_delta_zero_get
    if _newclass:rprop_delta_zero = property(_libfann.fann_rprop_delta_zero_get, _libfann.fann_rprop_delta_zero_set)
    __swig_setmethods__["train_slopes"] = _libfann.fann_train_slopes_set
    __swig_getmethods__["train_slopes"] = _libfann.fann_train_slopes_get
    if _newclass:train_slopes = property(_libfann.fann_train_slopes_get, _libfann.fann_train_slopes_set)
    __swig_setmethods__["prev_steps"] = _libfann.fann_prev_steps_set
    __swig_getmethods__["prev_steps"] = _libfann.fann_prev_steps_get
    if _newclass:prev_steps = property(_libfann.fann_prev_steps_get, _libfann.fann_prev_steps_set)
    __swig_setmethods__["prev_train_slopes"] = _libfann.fann_prev_train_slopes_set
    __swig_getmethods__["prev_train_slopes"] = _libfann.fann_prev_train_slopes_get
    if _newclass:prev_train_slopes = property(_libfann.fann_prev_train_slopes_get, _libfann.fann_prev_train_slopes_set)
    def __init__(self, *args):
        _swig_setattr(self, fann, 'this', _libfann.new_fann(*args))
        _swig_setattr(self, fann, 'thisown', 1)
    def __del__(self, destroy=_libfann.delete_fann):
        try:
            if self.thisown: destroy(self)
        except: pass

class fannPtr(fann):
    def __init__(self, this):
        _swig_setattr(self, fann, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, fann, 'thisown', 0)
        _swig_setattr(self, fann,self.__class__,fann)
_libfann.fann_swigregister(fannPtr)

class fann_train_data(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, fann_train_data, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, fann_train_data, name)
    def __repr__(self):
        return "<C fann_train_data instance at %s>" % (self.this,)
    __swig_setmethods__["errno_f"] = _libfann.fann_train_data_errno_f_set
    __swig_getmethods__["errno_f"] = _libfann.fann_train_data_errno_f_get
    if _newclass:errno_f = property(_libfann.fann_train_data_errno_f_get, _libfann.fann_train_data_errno_f_set)
    __swig_setmethods__["error_log"] = _libfann.fann_train_data_error_log_set
    __swig_getmethods__["error_log"] = _libfann.fann_train_data_error_log_get
    if _newclass:error_log = property(_libfann.fann_train_data_error_log_get, _libfann.fann_train_data_error_log_set)
    __swig_setmethods__["errstr"] = _libfann.fann_train_data_errstr_set
    __swig_getmethods__["errstr"] = _libfann.fann_train_data_errstr_get
    if _newclass:errstr = property(_libfann.fann_train_data_errstr_get, _libfann.fann_train_data_errstr_set)
    __swig_setmethods__["num_data"] = _libfann.fann_train_data_num_data_set
    __swig_getmethods__["num_data"] = _libfann.fann_train_data_num_data_get
    if _newclass:num_data = property(_libfann.fann_train_data_num_data_get, _libfann.fann_train_data_num_data_set)
    __swig_setmethods__["num_input"] = _libfann.fann_train_data_num_input_set
    __swig_getmethods__["num_input"] = _libfann.fann_train_data_num_input_get
    if _newclass:num_input = property(_libfann.fann_train_data_num_input_get, _libfann.fann_train_data_num_input_set)
    __swig_setmethods__["num_output"] = _libfann.fann_train_data_num_output_set
    __swig_getmethods__["num_output"] = _libfann.fann_train_data_num_output_get
    if _newclass:num_output = property(_libfann.fann_train_data_num_output_get, _libfann.fann_train_data_num_output_set)
    __swig_setmethods__["input"] = _libfann.fann_train_data_input_set
    __swig_getmethods__["input"] = _libfann.fann_train_data_input_get
    if _newclass:input = property(_libfann.fann_train_data_input_get, _libfann.fann_train_data_input_set)
    __swig_setmethods__["output"] = _libfann.fann_train_data_output_set
    __swig_getmethods__["output"] = _libfann.fann_train_data_output_get
    if _newclass:output = property(_libfann.fann_train_data_output_get, _libfann.fann_train_data_output_set)
    def __init__(self, *args):
        _swig_setattr(self, fann_train_data, 'this', _libfann.new_fann_train_data(*args))
        _swig_setattr(self, fann_train_data, 'thisown', 1)
    def __del__(self, destroy=_libfann.delete_fann_train_data):
        try:
            if self.thisown: destroy(self)
        except: pass

class fann_train_dataPtr(fann_train_data):
    def __init__(self, this):
        _swig_setattr(self, fann_train_data, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, fann_train_data, 'thisown', 0)
        _swig_setattr(self, fann_train_data,self.__class__,fann_train_data)
_libfann.fann_train_data_swigregister(fann_train_dataPtr)

class fann_error(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, fann_error, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, fann_error, name)
    def __repr__(self):
        return "<C fann_error instance at %s>" % (self.this,)
    __swig_setmethods__["errno_f"] = _libfann.fann_error_errno_f_set
    __swig_getmethods__["errno_f"] = _libfann.fann_error_errno_f_get
    if _newclass:errno_f = property(_libfann.fann_error_errno_f_get, _libfann.fann_error_errno_f_set)
    __swig_setmethods__["error_log"] = _libfann.fann_error_error_log_set
    __swig_getmethods__["error_log"] = _libfann.fann_error_error_log_get
    if _newclass:error_log = property(_libfann.fann_error_error_log_get, _libfann.fann_error_error_log_set)
    __swig_setmethods__["errstr"] = _libfann.fann_error_errstr_set
    __swig_getmethods__["errstr"] = _libfann.fann_error_errstr_get
    if _newclass:errstr = property(_libfann.fann_error_errstr_get, _libfann.fann_error_errstr_set)
    def __init__(self, *args):
        _swig_setattr(self, fann_error, 'this', _libfann.new_fann_error(*args))
        _swig_setattr(self, fann_error, 'thisown', 1)
    def __del__(self, destroy=_libfann.delete_fann_error):
        try:
            if self.thisown: destroy(self)
        except: pass

class fann_errorPtr(fann_error):
    def __init__(self, this):
        _swig_setattr(self, fann_error, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, fann_error, 'thisown', 0)
        _swig_setattr(self, fann_error,self.__class__,fann_error)
_libfann.fann_error_swigregister(fann_errorPtr)

FANN_TRAIN_INCREMENTAL = _libfann.FANN_TRAIN_INCREMENTAL
FANN_TRAIN_BATCH = _libfann.FANN_TRAIN_BATCH
FANN_TRAIN_RPROP = _libfann.FANN_TRAIN_RPROP
FANN_TRAIN_QUICKPROP = _libfann.FANN_TRAIN_QUICKPROP
FANN_ERRORFUNC_LINEAR = _libfann.FANN_ERRORFUNC_LINEAR
FANN_ERRORFUNC_TANH = _libfann.FANN_ERRORFUNC_TANH
FANN_LINEAR = _libfann.FANN_LINEAR
FANN_THRESHOLD = _libfann.FANN_THRESHOLD
FANN_THRESHOLD_SYMMETRIC = _libfann.FANN_THRESHOLD_SYMMETRIC
FANN_SIGMOID = _libfann.FANN_SIGMOID
FANN_SIGMOID_STEPWISE = _libfann.FANN_SIGMOID_STEPWISE
FANN_SIGMOID_SYMMETRIC = _libfann.FANN_SIGMOID_SYMMETRIC
FANN_SIGMOID_SYMMETRIC_STEPWISE = _libfann.FANN_SIGMOID_SYMMETRIC_STEPWISE
FANN_GAUSSIAN = _libfann.FANN_GAUSSIAN
FANN_GAUSSIAN_STEPWISE = _libfann.FANN_GAUSSIAN_STEPWISE
FANN_ELLIOT = _libfann.FANN_ELLIOT
FANN_ELLIOT_SYMMETRIC = _libfann.FANN_ELLIOT_SYMMETRIC

fann_run = _libfann.fann_run

fann_test = _libfann.fann_test

get_train_data_input = _libfann.get_train_data_input

get_train_data_output = _libfann.get_train_data_output

fann_is_NULL = _libfann.fann_is_NULL
cvar = _libfann.cvar
FANN_TRAIN_NAMES = cvar.FANN_TRAIN_NAMES
FANN_ERRORFUNC_NAMES = cvar.FANN_ERRORFUNC_NAMES
FANN_ACTIVATION_NAMES = cvar.FANN_ACTIVATION_NAMES

