{*------------------------------------------------------------------------------
   TFannNetwork encapsulates the Fast Artificial Neural Network.

   fann.sourceforge.net

   @author Maurício Pereira Maia
   @version 1.1

-------------------------------------------------------------------------------}
unit FannNetwork;

interface

{$R fann.dcr}


uses
  Windows, Messages, SysUtils, Classes,Fann;

type CardinalArray = array [0..1023] of Cardinal; //Array of Cardinal Type
     PCardinalArray = ^CardinalArray; //Pointer to CardinalArray Type


type
{*------------------------------------------------------------------------------
  TFannNetwork Component
-------------------------------------------------------------------------------}

  TFannNetwork = class(TComponent)
  private
    ann: PFann;
    pBuilt: boolean;
    pLayers: TStrings;
    pLearningRate: Single;
    pConnectionRate: Single;
    procedure SetLayers(const Value: TStrings);

  protected
    { Protected declarations }
  public
    { Public declarations }
    constructor Create(Aowner: TComponent); override;
    destructor Destroy(); override;
    procedure Build();
    function Train(Input: array of fann_type; Output: array of fann_type): single;
    procedure TrainOnFile(FileName: String; MaxEpochs: Cardinal; DesiredError: Single);
    procedure Run(Inputs: array of fann_type; var Outputs: array of fann_type);
    function GetMSE(): Single;
    procedure SaveToFile(FileName: String);
    procedure LoadFromFile(Filename: string);

  published

    {*------------------------------------------------------------------------------
     Network Layer Structure. Each line need to have the number of neurons
        of the layer.
        2
        4
        1
        Will make a three layered network with 2 input neurons, 4 hidden neurons
        and 1 output neuron.
   -------------------------------------------------------------------------------}
    property Layers: TStrings read PLayers write SetLayers;

    {*------------------------------------------------------------------------------
     Network Learning Rate.
    -------------------------------------------------------------------------------}
    property LearningRate: Single read pLearningRate write pLearningRate;
    {*------------------------------------------------------------------------------
     Network Connection Rate. See the FANN docs for more info.
    -------------------------------------------------------------------------------}
    property ConnectionRate: Single read pConnectionRate write pConnectionRate;
end;

procedure Register;

implementation

procedure Register; //Register the Component
begin
  RegisterComponents('FANN', [TFannNetwork]);
end;


{*------------------------------------------------------------------------------
    Builds a Fann neural nerwork.
    Layers property must be set to work.
-------------------------------------------------------------------------------}
procedure TFannNetwork.Build;
var l: PCardinalArray; //Pointer to number of neurons on each layer
    nl: integer; //Number of layers
    i: integer;
begin

        nl:=pLayers.Count;

        if nl=0 then
                Exception.Create('You need to specify the Layers property');

        GetMem(l,nl*sizeof(Cardinal)); //Alloc mem for the array


        for i:=0 to nl-1 do
        begin

                l[i]:=StrtoInt(pLayers.Strings[i]);
                
        end;

        ann:=fann_create_array(PConnectionRate,PLearningRate,nl,l[0]);
        fann_randomize_weights(ann,-0.5,0.5);

        pbuilt:=true;

        FreeMem(l);


end;

{*------------------------------------------------------------------------------
    Creates an instance of TFannNetwork

    @param      AOwner Owner Component
-------------------------------------------------------------------------------}
constructor TFannNetwork.Create(Aowner: TComponent);
begin
        inherited Create(Aowner);
        
        pBuilt:=false;
        pConnectionRate:=1;
        pLearningRate:=0.7;
        pLayers:=TStringList.Create;


end;

{*------------------------------------------------------------------------------
    Destroys the TFannNetwork object and frees its memory.
-------------------------------------------------------------------------------}
destructor TFannNetwork.Destroy;
begin
  If pBuilt then fann_destroy(ann);

  FreeAndNil(pLayers);
  inherited;
end;

{*------------------------------------------------------------------------------
    Returns the Mean Square Error of the Network.

    @return             Mean Square Error
-------------------------------------------------------------------------------}
function TFannNetwork.GetMSE: Single;
begin
        if not pBuilt then
               raise Exception.Create('The nework has not been built');

        result:=fann_get_mse(ann);

end;

{*------------------------------------------------------------------------------
    Loads a network from a file.

    @param  Filename    File that contains the network
    @see                SavetoFile
-------------------------------------------------------------------------------}
procedure TFannNetwork.LoadFromFile(Filename: string);
begin

        If pBuilt then fann_destroy(ann);

        ann:=fann_create_from_file(PChar(Filename));

        pBuilt:=true;

end;

{*------------------------------------------------------------------------------
    Executes the network.

    @param  Inputs      Array with the value of each input
    @param  Ouputs      Will receive the output of the network. Need to be
                        allocated before the Run function call with the number
                        of outputs on the network.
-------------------------------------------------------------------------------}
procedure TFannNetwork.Run(Inputs: array of fann_type;
  var Outputs: array of fann_type);
var O: Pfann_type_array;
    i: integer;
begin
        if not pBuilt then
               raise Exception.Create('The nework has not been built');

        O:=fann_run(ann,Inputs[0]);

        for i:=0 to High(outputs) do
        begin
            Outputs[i]:=O[i];
        end;

        

end;

{*------------------------------------------------------------------------------
    Saves the current network to the specified file.

    @param  Filename    Filename of the network

    @see                LoadFromFile
-------------------------------------------------------------------------------}
procedure TFannNetwork.SaveToFile(FileName: String);
begin
        if not pBuilt then exit;


        fann_save(ann,PChar(Filename));
        

end;

{*------------------------------------------------------------------------------
    Train the network and returns the Mean Square Error.

    @param  Input       Array with the value of each input of the network
    @param  Output      Array with the value of each desired output of the network
    @return             Mean Square Error after the training

    @see                TrainOnFile
-------------------------------------------------------------------------------}
procedure TFannNetwork.SetLayers(const Value: TStrings);
begin

   if Value.Count < 2 then
      raise Exception.Create('The network should have at least two layers')
   else
      pLayers.Assign(Value);

end;

{*------------------------------------------------------------------------------
    Will train one iteration with a set of inputs, and a set of desired outputs.

    @param  Input   Network inputs
    @param  Output  Network Desired Outputs

    @see                TrainOnFile
-------------------------------------------------------------------------------}
function TFannNetwork.Train(Input, Output: array of fann_type): single;
begin

        if not pBuilt then
                raise Exception.Create('The nework has not been built');

        fann_reset_mse(ann);
        fann_train(ann,Input[0],Output[0]);

        result:=fann_get_mse(ann);
        
end;

{*------------------------------------------------------------------------------
    Trains the network using the data in filename until desired error is reached
    or until maxepochs is surpassed.

    @param  FileName    Training Data File.
    @param  MaxEpochs   Maximum number of epochs to train
    @param  DesiredError Desired Error of the network after the training

    @see                Train
-------------------------------------------------------------------------------}
procedure TFannNetwork.TrainOnFile(FileName: String; MaxEpochs: Cardinal;
  DesiredError: Single);
begin

        if not pBuilt then
                raise Exception.Create('The nework has not been built');

        fann_train_on_file(ann,PChar(FileName),MaxEpochs,1000,DesiredError);

end;

end.
 