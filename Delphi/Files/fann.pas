unit fann;

interface

{$IFNDEF FIXEDFANN}
const DLL_FILE = 'fannfloat.dll';
{$ELSE}
const DLL_FILE = 'fannfixed.dll';
{$ENDIF}


{$IFDEF VER150}
   {$DEFINE VARIABLE_ARGUMENTS}
{$ENDIF}

{$IFDEF VER140}
   {$DEFINE VARIABLE_ARGUMENTS}
{$ENDIF}

//Only Delphi 6 and 7 supports the varargs directive


type


        fann_type = single;

        Pfann_type = ^fann_type;

        PPfann_type = ^pfann_type;

        fann_type_array = array [0..32767] of fann_type;

        Pfann_type_array = ^fann_type_array;

        (* MICROSOFT VC++ STDIO'S FILE DEFINITION*)
        _iobuf = packed record
                _ptr: Pchar;
                _cnt: integer;
                _base: Pchar;
                _flag: integer;
                _file: integer;
                _charbuf: integer;
                _bufsiz: integer;
                _tmpfname: Pchar;
        end;


        PFile = ^TFile;
        TFile = _iobuf;


        PPFann_Neuron = ^PFann_Neuron;
        PFann_Neuron = ^TFann_Neuron;
        TFann_Neuron = packed record
                weights: PFann_type;
                connected_neurons: PPFann_Neuron;
                num_connections: cardinal;
                value: fann_type;
        end;


        PFann_Layer = ^TFann_Layer;
        TFann_Layer = packed record
                first_neuron: PFann_Neuron;
                last_neuron: PFann_Neuron;
        end;

        PFann = ^TFann;
        TFann = packed record
                errno_f: cardinal;
                error_log: PFile;
                errstr: Pchar;
                learning_rate: single;
                connection_rate: single;
                shortcut_connections: cardinal;
                first_layer: PFann_Layer;
                last_layer: PFann_Layer;
                total_neurons: cardinal;
                num_input: cardinal;
                num_output: cardinal;
                train_errors: Pfann_type;
                activation_function_hidden,activation_function_output: cardinal;
                activation_steepness_hidden: fann_type;
                activation_steepness_output: fann_type;
                training_algorithm: cardinal;
                {$IFDEF FIXEDFANN}
                 decimal_point: cardinal;
                 multiplier: cardinal;
                {$ENDIF}
                activation_results_hidden: array [0..5] of fann_type;
                activation_values_hidden: array [0..5] of fann_type;
                activation_results_output: array [0..5] of fann_type;
                activation_values_output: array [0..5] of fann_type;

                total_connections: cardinal;
                output: pfann_type;
                num_MSE: cardinal;
                MSE_value: single;
                train_error_function: cardinal;
                quickprop_decay: single;
                quickprop_mu: single;
                rprop_increase_factor: single;
                rprop_decrease_factor: single;
                rprop_delta_min: single;
                rprop_delta_max: single;
                train_slopes: pfann_type;
                prev_steps: pfann_type;
                prev_train_slopes: pfann_type;
        end;

        PFann_Train_Data = ^TFann_Train_Data;
        TFann_Train_Data = packed record
                errno_f: cardinal;
                erro_log: PFile;
                errstr: Pchar;
                num_data: cardinal;
                num_input: cardinal;
                num_ouput: cardinal;
                input: PPFann_Type;
                output: PPFann_Type;
        end;

        PFann_Error = ^TFann_Error;
        TFann_Error = packed record
                error_log: PFile;
                errstr: PChar;
        end;

        _Fann_Train =
        (
          	FANN_TRAIN_INCREMENTAL = 0,
	        FANN_TRAIN_BATCH,
	        FANN_TRAIN_RPROP,
                FANN_TRAIN_QUICKPROP
        );

        _Fann_Error_Func =
        (
                FANN_ERRORFUNC_LINEAR = 0,
	        FANN_ERRORFUNC_TANH
        );

        TFann_Report_CallBack = function(epochs: cardinal;error: single): integer; stdcall;

var
        FANN_ERRORFUNC_NAMES: array [0..1] of string = (
        	'FANN_ERRORFUNC_LINEAR',
	        'FANN_ERRORFUNC_TANH'
        );

        FANN_TRAIN_NAMES: array [0..3] of string =
        (
	        'FANN_TRAIN_INCREMENTAL',
	        'FANN_TRAIN_BATCH',
	        'FANN_TRAIN_RPROP',
	        'FANN_TRAIN_QUICKPROP'
        );

        {$IFDEF VARIABLE_ARGUMENTS}

        function fann_create(connection_rate: single; learning_rate: single;
                                num_layers: cardinal): PFann; cdecl; varargs;

        function fann_create_shortcut(learning_rate: single;num_layers: cardinal): PFann; cdecl; varargs;

        {$ENDIF}

        function fann_create_array(connection_rate: single; learning_rate: single;
                                num_layers: cardinal;var layers: cardinal): PFann; stdcall;



        function fann_create_shortcut_array(learning_rate: single; num_layers: cardinal;var layers: cardinal): PFann; stdcall;

        function fann_run(ann: PFann;var input: Fann_Type): Pfann_type_array; stdcall;

        procedure fann_destroy(Ann: PFann); stdcall;

        procedure fann_randomize_weights(Ann: PFann; Min_weight: fann_type; Max_weight: fann_type); stdcall;

        procedure fann_init_weights(Ann: PFann;var train_data: TFann_Train_Data); stdcall;

        procedure fann_print_connections(ann: PFann);stdcall;

        function fann_create_from_file(const configuration_file: PChar): PFann; stdcall;

        procedure fann_save(Ann: PFann; Const Configuration_File: PChar);stdcall;

        function fann_save_to_fixed(Ann: PFann; Const Configuration_File: PChar): integer;stdcall;

        {$IFNDEF FIXEDFANN}
                procedure fann_train(Ann: PFann;var Input: Fann_Type;var Desired_Output: Fann_Type);stdcall;
        {$ENDIF}

        function fann_test(Ann: PFann; var Input: Fann_Type; var Desired_Output: fann_Type): Pfann_type_array;stdcall;

        function fann_get_MSE(Ann: PFann): single;stdcall;

        procedure fann_reset_MSE(Ann: Pfann); stdcall;

        function fann_read_train_from_file(filename: PChar): PFann_Train_Data; stdcall;

        procedure fann_destroy_train(var Train_Data: TFann_Train_Data);stdcall;

        {$IFNDEF FIXEDFANN}
              function fann_train_epoch(Ann: PFann;var data: TFann_Train_Data): single;
              function fann_test_data(Ann: PFann;var data: TFann_Train_Data): single;
              procedure fann_train_on_data(Ann: PFann; var Data: TFann_Train_Data;max_epochs: cardinal;epochs_between_reports: cardinal; desired_error: single);stdcall;

              procedure fann_train_on_data_callback(Ann: PFann; var Data: TFann_Train_Data; max_epochs: cardinal;epochs_between_reports: cardinal; desired_error: single;CallBack: TFann_Report_Callback); stdcall;

              procedure fann_train_on_file(Ann: PFann; Filename: Pchar;max_epochs: cardinal;epochs_between_reports: cardinal; desired_error: single); stdcall;

              procedure fann_train_on_file_callback(Ann: PFann; Filename: Pchar;max_epochs: cardinal;epochs_between_reports: cardinal; desired_error: single; CallBack: TFann_Report_Callback);stdcall;

              procedure fann_shuffle_train_data(var Train_Data: TFann_Train_Data);stdcall;

              function fann_merge_train_data(var Data1: TFann_Train_Data;var Data2: TFann_Train_Data): PFann_Train_Data; stdcall;

              function fann_duplicate_train_data(var Data: TFann_Train_Data): PFann_Train_Data;stdcall;

        {$ENDIF}

        procedure fann_save_train(var Data: TFann_train_Data; Filename: PChar);stdcall;

        procedure fann_save_train_to_fixed(var Data: TFann_train_Data; FileName: Pchar; decimal_point: cardinal);stdcall;

        procedure fann_print_parameters(Ann: PFann); stdcall;

        function fann_get_training_algorithm(Ann: Pfann): cardinal;stdcall;

        procedure fann_set_training_algorithm(Ann: PFann; Training_Algorithm: cardinal);stdcall;

        function fann_get_learning_rate(Ann: PFann): single;stdcall;

        procedure fann_set_learning_rate(Ann: PFann; Learning_Rate: Single); stdcall;

        function fann_get_activation_function_hidden(Ann: PFann): cardinal;stdcall;

        procedure fann_set_activation_function_hidden(Ann: Pfann; Activation_function: cardinal); stdcall;

        function fann_get_activation_function_output(Ann: Pfann): cardinal;stdcall;

        procedure fann_set_activation_function_output(Ann: Pfann; Activation_Function: cardinal); stdcall;

        function fann_get_activation_steepness_hidden(Ann: PFann): fann_type; stdcall;

        procedure fann_set_activation_steepness_hidden(Ann: PFann; SteepNess: Fann_Type); stdcall;

        function fann_get_activation_steepness_output(Ann: PFann): fann_type;stdcall;

        procedure fann_set_activation_steepness_output(Ann: PFann; SteepNess: Fann_Type);stdcall;

        procedure fann_set_train_error_function(Ann: PFann; Train_Error_Function: cardinal); stdcall;

        function fann_get_train_error_function(Ann: PFann): cardinal;stdcall;

        function fann_get_quickprop_decay(Ann: PFann): single;stdcall;

        procedure fann_set_quickprop_decay(Ann: Pfann; quickprop_decay: Single);stdcall;

        function fann_get_quickprop_mu(Ann: PFann): single;stdcall;

        procedure fann_set_quickprop_mu(Ann: PFann; Mu: Single);stdcall;

        function fann_get_rprop_increase_factor(Ann: PFann): single;stdcall;

        procedure fann_set_rprop_increase_factor(Ann: PFann;rprop_increase_factor: single);stdcall;

        function fann_get_rprop_decrease_factor(Ann: PFann): single;stdcall;

        procedure fann_set_rprop_decrease_factor(Ann: PFann;rprop_decrease_factor: single); stdcall;

        function fann_get_rprop_delta_min(Ann: PFann): single; stdcall;

        procedure fann_set_rprop_delta_min(Ann: PFann; rprop_delta_min: Single); stdcall;

        function fann_get_rprop_delta_max(Ann: PFann): single;stdcall;

        procedure fann_set_rprop_delta_max(Ann: PFann; rprop_delta_max: Single); stdcall;

        function fann_get_num_input(Ann: PFann): cardinal;stdcall;

        function fann_get_num_output(Ann: PFann): cardinal;stdcall;

        function fann_get_total_neurons(Ann: PFann): cardinal; stdcall;

        function fann_get_total_connections(Ann: PFann): cardinal; stdcall;

        {$IFDEF FIXEDFANN}

                function fann_get_decimal_point(Ann: Pfann): cardinal; stdcall;

                function fann_get_multiplier(Ann: PFann): cardinal;stdcall;

        {$ENDIF}

        procedure fann_set_error_log(errdat: PFann_Error; Log_File: PFile);stdcall;

        function fann_get_errno(errdat: PFann_Error): cardinal;stdcall;

        procedure fann_reset_errno(errdat: PFann_Error);stdcall;

        procedure fann_reset_errstr(errdat: PFann_Error);stdcall;

        function fann_get_errstr(errdat: PFann_Error): PChar;stdcall;

        procedure fann_print_error(Errdat: PFann_Error);stdcall;


implementation

        {$IFDEF VARIABLE_ARGUMENTS}
        function fann_create;external DLL_FILE;

        function fann_create_shortcut;external DLL_FILE;
        {$ENDIF}
        
        
        function fann_create_array;external DLL_FILE name '_fann_create_array@16';



        function fann_create_shortcut_array;external DLL_FILE name '_fann_create_shortcut_array@12';

        function fann_run;external DLL_FILE name '_fann_run@8';

        procedure fann_destroy;external DLL_FILE name '_fann_destroy@4';

        procedure fann_randomize_weights;external DLL_FILE name '_fann_randomize_weights@12';

        procedure fann_init_weights;external DLL_FILE name '_fann_init_weights@8';

        procedure fann_print_connections;external DLL_FILE name '_fann_print_connections@4';

        function fann_create_from_file;external DLL_FILE name '_fann_create_from_file@4';

        procedure fann_save;external DLL_FILE name '_fann_save@8';

        function fann_save_to_fixed;external DLL_FILE name '_fann_save_to_fixed@8';

        {$IFNDEF FIXEDFANN}
                procedure fann_train;external DLL_FILE name '_fann_train@12';
        {$ENDIF}

        function fann_test;external DLL_FILE name '_fann_test@12';

        function fann_get_MSE;external DLL_FILE name '_fann_get_MSE@4';

        procedure fann_reset_MSE;external DLL_FILE name '_fann_reset_MSE@4';

        function fann_read_train_from_file;external DLL_FILE name '_fann_read_train_from_file@4';

        procedure fann_destroy_train;external DLL_FILE name '_fann_destroy_train@4';

        {$IFNDEF FIXEDFANN}
              function fann_train_epoch;external DLL_FILE name '_fann_train_epoch@8';

              function fann_test_data;external DLL_FILE name '_fann_test_data@8';

              procedure fann_train_on_data;external DLL_FILE name '_fann_train_on_data@20';

              procedure fann_train_on_data_callback;external DLL_FILE name '_fann_train_on_data_callback@24';

              procedure fann_train_on_file;external DLL_FILE name '_fann_train_on_file@20'

              procedure fann_train_on_file_callback;external DLL_FILE name '_fann_train_on_file_callback@24';

              procedure fann_shuffle_train_data;external DLL_FILE name '_fann_shuffle_train_data@4';

              function fann_merge_train_data;external DLL_FILE name '_fann_merge_train_data@8';

              function fann_duplicate_train_data;external DLL_FILE name '_fann_duplicate_train_data@4';

        {$ENDIF}

        procedure fann_save_train;external DLL_FILE name '_fann_save_train@8';

        procedure fann_save_train_to_fixed;external DLL_FILE name '_fann_save_train_to_fixed@12';

        procedure fann_print_parameters;external DLL_FILE name '_fann_print_parameters@4';

        function fann_get_training_algorithm;external DLL_FILE name '_fann_get_training_algorithm@4';

        procedure fann_set_training_algorithm;external DLL_FILE name '_fann_set_training_algorithm@8';

        function fann_get_learning_rate;external DLL_FILE name '_fann_get_learning_rate@4';

        procedure fann_set_learning_rate;external DLL_FILE name '_fann_set_learning_rate@8';

        function fann_get_activation_function_hidden;external DLL_FILE name '_fann_get_activation_function_hidden@4';

        procedure fann_set_activation_function_hidden;external DLL_FILE name '_fann_set_activation_function_hidden@8';

        function fann_get_activation_function_output;external DLL_FILE name '_fann_get_activation_function_output@4';

        procedure fann_set_activation_function_output;external DLL_FILE name '_fann_set_activation_function_output@8';

        function fann_get_activation_steepness_hidden;external DLL_FILE name '_fann_get_activation_steepness_hidden@4';

        procedure fann_set_activation_steepness_hidden;external DLL_FILE name '_fann_set_activation_steepness_hidden@8';

        function fann_get_activation_steepness_output;external DLL_FILE name '_fann_get_activation_steepness_output@4';

        procedure fann_set_activation_steepness_output;external DLL_FILE name '_fann_set_activation_steepness_output@8';

        procedure fann_set_train_error_function;external DLL_FILE name '_fann_set_train_error_function@8';

        function fann_get_train_error_function;external DLL_FILE name '_fann_get_train_error_function@4';

        function fann_get_quickprop_decay;external DLL_FILE name '_fann_get_quickprop_decay@4';

        procedure fann_set_quickprop_decay;external DLL_FILE name '_fann_set_quickprop_decay@8';

        function fann_get_quickprop_mu;external DLL_FILE name '_fann_get_quickprop_mu@4';

        procedure fann_set_quickprop_mu;external DLL_FILE name '_fann_set_quickprop_mu@8';

        function fann_get_rprop_increase_factor;external DLL_FILE name '_fann_get_rprop_increase_factor@4';

        procedure fann_set_rprop_increase_factor;external DLL_FILE name '_fann_set_rprop_increase_factor@8';

        function fann_get_rprop_decrease_factor;external DLL_FILE name '_fann_get_rprop_decrease_factor@4';

        procedure fann_set_rprop_decrease_factor;external DLL_FILE name '_fann_set_rprop_decrease_factor@8';

        function fann_get_rprop_delta_min;external DLL_FILE name '_fann_get_rprop_delta_min@4';

        procedure fann_set_rprop_delta_min;external DLL_FILE name '_fann_set_rprop_delta_min@8';

        function fann_get_rprop_delta_max;external DLL_FILE name '_fann_get_rprop_delta_max@4';

        procedure fann_set_rprop_delta_max;external DLL_FILE name '_fann_set_rprop_delta_max@8';

        function fann_get_num_input;external DLL_FILE name '_fann_get_num_input@4';

        function fann_get_num_output;external DLL_FILE name '_fann_get_num_output@4';

        function fann_get_total_neurons;external DLL_FILE name '_fann_get_total_neurons@4';

        function fann_get_total_connections;external DLL_FILE name '_fann_get_total_connections@4';

        {$IFDEF FIXEDFANN}

                function fann_get_decimal_point;external DLL_FILE name '_fann_get_decimal_point@4';
                function fann_get_multiplier;external DLL_FILE name '_fann_get_decimal_point@4';

        {$ENDIF}

        procedure fann_set_error_log;external DLL_FILE name '_fann_set_error_log@8';

        function fann_get_errno;external DLL_FILE name '_fann_get_errno@4';

        procedure fann_reset_errno;external DLL_FILE name '_fann_reset_errno@4';

        procedure fann_reset_errstr;external DLL_FILE name '_fann_reset_errstr@4';

        function fann_get_errstr;external DLL_FILE name '_fann_get_errstr@4';

        procedure fann_print_error;external DLL_FILE name '_fann_print_error@4';

end.
