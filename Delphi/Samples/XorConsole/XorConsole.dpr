program XorConsole;

{$APPTYPE CONSOLE}

uses
  SysUtils,fann;

var ann: PFann;
    inputs: array [0..1] of fann_type;
    calc_out: PFann_Type_array;
    i,j: integer;

begin
        ann:=fann_create(1,0.7,3,2,4,1);


        fann_train_on_file(ann, 'xor.data', 500000,
                1000, 0.00001);

        fann_save(ann, 'xor_float.net');

        for i:=0 to 1 do
                for j:=0 to 1 do
                begin
                        inputs[0]:=j;
                        inputs[1]:=i;
                        calc_out:= fann_run(ann, inputs[0]);
                        writeln(Format('%d Xor %d = %f',[i,j,Calc_Out[0]]));
                end;
        readln;

        fann_destroy(ann);

end.
