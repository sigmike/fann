#!/usr/bin/python
import os
import sys
from pyfann import libfann
import signal, random
from optparse import OptionParser
random.seed()

class Trainer:
    def __init__(self, network_save_file):
        # initialize network parameters
        self.network_save_file = network_save_file



    def load_data(self, data_file,val_file=None):
        # create training data, and ann object
        print "Loading data"
        self.train_data = libfann.training_data()
        self.train_data.read_train_from_file(data_file)
        self.dim_input=self.train_data.num_input_train_data()
        self.dim_output=self.train_data.num_output_train_data()

        input=self.train_data.get_input()
        target=self.train_data.get_output()
        
        data_lo_hi=[0,0]
        for i in range(len(input)):
            if target[i][0]<0.5:
               data_lo_hi[0]+=1 
            elif target[i][0]>0.5:
               data_lo_hi[1]+=1
        
        print "\t Train data is %d low and %d high" % tuple(data_lo_hi)

        
        if (val_file and os.path.exists(val_file)):
            print "Loading validation data"
            self.do_validation=True
            self.val_data=libfann.training_data()
            self.val_data.read_train_from_file(val_file)
            input=self.val_data.get_input()
            target=self.val_data.get_output()
            data_lo_hi=[0,0]
            for i in range(len(input)):
                if target[i][0]<0.5:
                   data_lo_hi[0]+=1 
                elif target[i][0]>0.5:
                   data_lo_hi[1]+=1
            print "\t Validation data is %d low and %d high" % tuple(data_lo_hi)
        else:
            self.val_data=self.train_data
            self.do_validation=False


    def create_ann(self, network_load_file=None, connection_rate=1, learning_rate=0.2, num_neurons_hidden=15, config_file=None):
        self.connection_rate = connection_rate 
        self.learning_rate = learning_rate 
        self.num_neurons_hidden = num_neurons_hidden 
        
        if config_file:
            execfile(config_file)
        
        self.ann = libfann.neural_net()
        
        if (network_load_file and os.path.exists(network_load_file)):
            print "Loading network"	
            self.ann.create_from_file(network_load_file)
        else:
            print "Creating network"	
            self.ann.create_sparse_array(self.connection_rate, (self.dim_input, self.num_neurons_hidden, self.dim_output))


            self.ann.set_activation_function_hidden(libfann.SIGMOID)
            #self.ann.set_activation_function_hidden(libfann.SIGMOID_SYMMETRIC)
            #self.ann.set_activation_function_hidden(libfann.SIGMOID_STEPWISE)
            #self.ann.set_activation_function_hidden(libfann.SIGMOID_SYMMETRIC_STEPWISE)
            
            self.ann.set_activation_function_output(libfann.SIGMOID)
            #self.ann.set_activation_function_target(libfann.SIGMOID_SYMMETRIC)
            #self.ann.set_activation_function_target(libfann.SIGMOID_STEPWISE)
            #self.ann.set_activation_function_target(libfann.SIGMOID_SYMMETRIC_STEPWISE)
            
            self.ann.set_train_error_function(libfann.ERRORFUNC_TANH)
            #self.ann.set_train_error_function(libfann.ERRORFUNC_LINEAR)

            #self.ann.set_training_algorithm(libfann.TRAIN_INCREMENTAL)
            self.ann.set_training_algorithm(libfann.TRAIN_RPROP)
            #self.ann.set_training_algorithm(libfann.TRAIN_QUICKPROP)

            print "initializing weights"
            #self.ann.init_weights(self.train_data)
            #self.ann.randomize_weights(-.10,.10)
            
        self.ann.set_learning_rate(self.learning_rate)
        self.ann.set_rprop_increase_factor(1.2)
        self.ann.set_rprop_decrease_factor(0.5)
        self.ann.set_rprop_delta_min(0)
        self.ann.set_rprop_delta_max(50)
        self.ann.print_parameters()


    def train(self, desired_error = 0.000001, 
            max_iterations = 1000, iterations_between_reports = 10, error_center=0.5,):
        # start training the network
        self.desired_error = desired_error 
        self.max_iterations = max_iterations 
        self.iterations_between_reports = iterations_between_reports 
        self.error_center = error_center
        
        if self.do_validation:
            val_input=self.val_data.get_input()
            val_target=self.val_data.get_output()
        else:
            val_input=self.train_data.get_input()
            val_target=self.train_data.get_output()

        
        self.ann.set_bit_fail_limit(self.error_center)
        
        print "Training network"
        epoch=0
        last_best=0
        last_best_err=0
        exit_cond=False
        last_best_err=-1
        best_bit_fail="N/A"
        err_tr="N/A"
        err=0
        while not exit_cond:
            try:
                if self.do_validation:
                    err=self.ann.test_data(self.val_data)
                    
                if err<last_best_err:
                    self.ann.save(network_save_file)
                    last_best=epoch
                    last_best_err=err
                    if self.do_validation:
                        err,bf=self.test(val_input, val_target, self.error_center)
                        best_bit_fail=bf

                
                if (not epoch % self.iterations_between_reports):
                    if self.do_validation:
                        print "Epochs: %d, CTErr: %s, CVErr: %f, BVErr: %f, BitF[#err,[#elo,#ehi]]: %s, LSEp: %d" % \
                            (epoch, str(err_tr), err, last_best_err, str(best_bit_fail),last_best)
                    else:
                        print "Epochs: %d, CTErr: %f, BitF: %d" % \
                            (epoch, err_tr, self.ann.get_bit_fail())
                            
                err_tr=self.ann.train_epoch(self.train_data)
                if not self.do_validation:
                    err=err_tr
                if epoch==0:
                    last_best_err=err
                    
                epoch+=1

            except KeyboardInterrupt:
                self.ann.save(self.network_save_file+".current")
                exit_cond=True

    
    def test(self, input=None, target=None, error_center=0.5):
        if not input or not target:
            input=self.val_data.get_input()
            target=self.val_data.get_output()

        errors_lo_hi=[0,0]
        self.ann.reset_MSE()
        for i in range(len(input)):
            out=self.ann.test(input[i], target[i])
            if target[i][0]<0.5 and out[0]>error_center:
               errors_lo_hi[0]+=1 
            elif target[i][0]>0.5 and out[0]<error_center:
               errors_lo_hi[1]+=1 
        
        return  self.ann.get_MSE(), [sum(errors_lo_hi), errors_lo_hi]
        #return  tot_err/len(input), [sum(errors_lo_hi), errors_lo_hi, data_lo_hi]



def search_file(filename):
    if not (filename and os.path.exists(filename)):
        if filename:
            print"""\nFile %s could not be found\n""" % filename
        return False
    return True

if __name__ == "__main__":

    parser = OptionParser("usage: %prog -s net_file -d data_file [-l net_load_file]")

    parser.add_option("-d", "--data-file", type="string", dest="data_file", 
                help="Read data from the given file", default=None)

    parser.add_option("-l", "--network-load", type="string", dest="network_load_file", 
                help="Read fann_network data from the given file", default=None)

    parser.add_option("-s", "--network-save", type="string", dest="network_save_file", 
                help="Save fann_network data from the given file", default=None)
    
    parser.add_option("-c", "--config-file", type="string", dest="config_file", 
                help="Load config from the given file", default=None)

    parser.add_option("-v", "--validation-data", type="string", dest="val_file", 
                help="Save fann_network data from the given file", default=None)

    parser.add_option("-t", "--just-test", action="store_true", dest="just_test", 
                help="Do not train", default=False)

    parser.add_option("-r", "--reset-file", action="store_true", dest="reset_file", 
                help="Do not load the network from save train file", default=False)

    (options, args) = parser.parse_args(sys.argv[1:])

    data_file=options.data_file
    network_save_file=options.network_save_file
    network_load_file=options.network_load_file
    val_file=options.val_file
    just_test=options.just_test
    reset_file=options.reset_file

    if (not network_load_file and not reset_file):
        network_load_file=network_save_file


    if (not (search_file(data_file) and ( network_save_file or (just_test and search_file(network_load_file))))):
        parser.print_help()
        print data_file, network_save_file, network_load_file 
        parser.error("Incorrect number of arguments" + str(options))
        sys.exit(1)

    trainer=Trainer(network_save_file)
    trainer.load_data(data_file, val_file)
    trainer.create_ann(network_load_file,config_file=options.config_file)
    if not just_test:
        trainer.train()
    print "Testing network"
    trainer.create_ann(network_save_file,config_file=options.config_file)
    err,bf=trainer.test()
    print "Errors:  ", bf, err
    sys.exit(0)

    #signal.signal(signal.SIGINT,signal.SIG_DFL)
