object Form1: TForm1
  Left = 192
  Top = 107
  Width = 276
  Height = 170
  Caption = 'Delphi FANN Xor Demo'
  Color = clBtnFace
  Font.Charset = DEFAULT_CHARSET
  Font.Color = clWindowText
  Font.Height = -11
  Font.Name = 'MS Sans Serif'
  Font.Style = []
  OldCreateOrder = False
  PixelsPerInch = 96
  TextHeight = 13
  object LblError: TLabel
    Left = 104
    Top = 36
    Width = 111
    Height = 13
    Caption = 'Mean Square Error:'
    Font.Charset = DEFAULT_CHARSET
    Font.Color = clWindowText
    Font.Height = -11
    Font.Name = 'MS Sans Serif'
    Font.Style = [fsBold]
    ParentFont = False
  end
  object lblMSE: TLabel
    Left = 216
    Top = 36
    Width = 5
    Height = 13
    Font.Charset = DEFAULT_CHARSET
    Font.Color = clRed
    Font.Height = -11
    Font.Name = 'MS Sans Serif'
    Font.Style = [fsBold]
    ParentFont = False
  end
  object btnTrain: TButton
    Left = 0
    Top = 32
    Width = 97
    Height = 25
    Caption = 'Train'
    Enabled = False
    TabOrder = 0
    OnClick = btnTrainClick
  end
  object btnRun: TButton
    Left = 0
    Top = 64
    Width = 97
    Height = 25
    Caption = 'Run'
    Enabled = False
    TabOrder = 1
    OnClick = btnRunClick
  end
  object memoXor: TMemo
    Left = 104
    Top = 64
    Width = 113
    Height = 73
    Lines.Strings = (
      '')
    TabOrder = 2
  end
  object btnBuild: TButton
    Left = 0
    Top = 0
    Width = 97
    Height = 25
    Caption = 'Build'
    TabOrder = 3
    OnClick = btnBuildClick
  end
  object NN: TFannNetwork
    Layers.Strings = (
      '2'
      '4'
      '1')
    LearningRate = 0.699999988079071
    ConnectionRate = 1
    Top = 104
  end
end
