#!/usr/bin/perl
use Data::Dumper;
use POSIX;
use Math::BigFloat;

@algorithms = ({ALGO => 'fann_cascade_rprop_one_activation', NAME => 'Cascade2 RPROP One'},
	       {ALGO => 'fann_cascade_rprop_multi_activation', NAME => 'Cascade2 RPROP Multi'},
	       {ALGO => 'fann_cascade_quickprop_one_activation', NAME => 'Cascade2 Quickprop One'},
	       {ALGO => 'fann_cascade_quickprop_multi_activation', NAME => 'Cascade2 Quickprop Multi'},
	       {ALGO => 'fann_rprop', NAME => 'iRPROP$^{-}$'},
	       {ALGO => 'fann_quickprop', NAME => 'Quickprop'},
	       {ALGO => 'fann_batch', NAME => 'Batch'},
	       {ALGO => 'fann_incremental_momentum', NAME => 'Incremental'},
	       {ALGO => 'lwnn', NAME => '(External) Lwnn Incremental'},
	       {ALGO => 'jneural', NAME => '(External) Jneural Incremental'});
 
$oldDataSet = "";
$numDataSet = 0;

while(<>)
{
    chop;
    @data = split / /;
    @words = split(/\./, @data[0]);
    $dataSet = @words[0];
    $algoritm = @words[1];
    $test_train = @words[2];
    $mse = Math::BigFloat->new(@data[2]);

    if($oldDataSet ne $dataSet && $oldDataSet ne "")
    {
	$numDataSet++;
	print "**$oldDataSet**\n";
	@test = keys(%summary);

	$min_test_mse = 0;
	$min_train_mse = 0;
	#Adds rank to each set and calculates percent etc.
	for($i = 1; $i <= scalar(%summary)+2; $i++)
	{
	    $min_algo_test = 0;
	    $min_algo_train = 0;

	    while(($algo, $foo) = each(%summary))
	    {
		if(!exists($foo->{'test_rank'}) && (!$min_algo_test || $foo->{'test'} < $summary{$min_algo_test}->{'test'}))
		{
		    $min_algo_test = $algo;
		    #print "Setting min test algo ", $min_algo_test, " -> ", $foo->{'test'}, "\n";
		}
		
		if(!exists($foo->{'train_rank'}) && (!$min_algo_train || $foo->{'train'} < $summary{$min_algo_train}->{'train'}))
		{
		    $min_algo_train = $algo;
		    #print "Setting min train algo ", $min_algo_train, " -> ", $foo->{'train'}, "\n";
		}
	    }

	    $summary{$min_algo_test}->{'test_rank'} = $i;
	    $summary{$min_algo_train}->{'train_rank'} = $i;
	    if($i == 1)
	    {
		$summary{$min_algo_test}->{'test_percent'} = Math::BigFloat->new(100);
		$summary{$min_algo_train}->{'train_percent'} = Math::BigFloat->new(100);
		$min_test_mse = $summary{$min_algo_test}->{'test'};
		$min_train_mse = $summary{$min_algo_train}->{'train'};
	    }
	    else
	    {
		$summary{$min_algo_test}->{'test_percent'} = ($min_test_mse*100)/$summary{$min_algo_test}->{'test'};
		#print $summary{$min_algo_test}->{'test_percent'}, " = ", $min_test_mse, "/", $summary{$min_algo_test}->{'test'}, ";\n";
		$summary{$min_algo_train}->{'train_percent'} = ($min_train_mse*100)/$summary{$min_algo_train}->{'train'};
		#print $summary{$min_algo_train}->{'train_percent'}, " = ", $min_train_mse, "/", $summary{$min_algo_train}->{'train'}, ";\n";
	    }
	}

	#print Dumper(%summary);

	foreach $algo (@algorithms)
	{
	    $foo = $summary{$algo->{ALGO}};
	    printf("%s & %.8f & %.2f & %d & %.8f & %.2f & %d\n", 
		   $algo->{NAME}, 
		   $foo->{'train'}, $foo->{'train_percent'}, $foo->{'train_rank'}, 
		   $foo->{'test'}, $foo->{'test_percent'}, $foo->{'test_rank'});

	    $total{$algo->{ALGO}}->{'train'} += $foo->{'train'};
	    $total{$algo->{ALGO}}->{'test'} += $foo->{'test'};
	    $total{$algo->{ALGO}}->{'train_percent'} += $foo->{'train_percent'};
	    $total{$algo->{ALGO}}->{'test_percent'} += $foo->{'test_percent'};
	    $total{$algo->{ALGO}}->{'train_rank'} += $foo->{'train_rank'};
	    $total{$algo->{ALGO}}->{'test_rank'} += $foo->{'test_rank'};
	}

	undef(%summary);
    }

    $summary{$algoritm}->{$test_train} = $mse;
    $oldDataSet = $dataSet;
}

print "**Average**\n";
foreach $algo (@algorithms)
{
    $foo = $total{$algo->{ALGO}};
    printf("%s & %.6f & %.2f & %.2f & %.6f & %.2f & %.2f\n", 
	   $algo->{NAME}, 
	   $foo->{'train'}/$numDataSet, $foo->{'train_percent'}/$numDataSet, $foo->{'train_rank'}/$numDataSet, 
	   $foo->{'test'}/$numDataSet, $foo->{'test_percent'}/$numDataSet, $foo->{'test_rank'}/$numDataSet);
}
