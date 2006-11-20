#!/usr/bin/perl
use Data::Dumper;
use POSIX;
use Math::BigFloat;

$path = "/home/lukesky/rl-thesis/doc/";

@algorithms = ({ALGO => 'fann_cascade_rprop_one_activation', NAME => 'C2 RPROP Single'},
	       {ALGO => 'fann_cascade_rprop_multi_activation', NAME => 'C2 RPROP Multi'},
	       {ALGO => 'fann_cascade_quickprop_one_activation', NAME => 'C2 Quickprop Single'},
	       {ALGO => 'fann_cascade_quickprop_multi_activation', NAME => 'C2 Quickprop Multi'},
	       {ALGO => 'fann_rprop', NAME => 'iRPROP$^{-}$'},
	       {ALGO => 'fann_quickprop', NAME => 'Quickprop'},
	       {ALGO => 'fann_batch', NAME => 'Batch'},
	       {ALGO => 'fann_incremental_momentum', NAME => 'Incremental'},
	       {ALGO => 'lwnn', NAME => 'Lwnn Incr.'},
	       {ALGO => 'jneural', NAME => 'Jneural Incr.'}
);

#This is not used yet
%dataSetType = {'abelone' => 'R',
		'bank32fm' => 'R',
		'bank32nh' => 'R',
		'kin32fm' => 'R',
		'census-house' => 'R',
		'building' => 'R',
		'diabetes' => 'C',
		'gene' => 'C',
		'mushroom' => 'C',
		'parity8' => 'C',
		'parity13' => 'C',
		'pumadyn-32fm' => 'R',
		'robot' => 'R',
		'soybean' => 'C',
		'thyroid' => 'C',
		'two-spiral' => 'C'};
		

$oldDataSet = "";
$numDataSet = 0;

while(<>)
{
    chop;
    @data = split / /;
    @words = split(/\./, @data[0]);
    $dataSet = @words[0];
    $algorithm = @words[1];
    $test_train = @words[2];
    $mse = Math::BigFloat->new(@data[2]);
    $meanbitfail = Math::BigFloat->new(@data[5]);
    
    $found_algo = 0;
	foreach $algo (@algorithms)
	{
		if($algo->{ALGO} eq $algorithm)
		{
			$found_algo = 1;
		}
	}

	#when we change dataset, then print table
	if($oldDataSet ne $dataSet && $oldDataSet ne "")
	{
		$numDataSet++;
		print "**$oldDataSet**\n";
	
		#calculate min and max mse
		$min_test_mse = 100;
		$min_train_mse = 100;
		$max_test_mse = -100;
		$max_train_mse = -100;
		while(($algo, $foo) = each(%summary))
		{
			if($foo->{'test'} < $min_test_mse)
			{
				$min_test_mse = $foo->{'test'};
			}
			if($foo->{'train'} < $min_train_mse)
			{
				$min_train_mse = $foo->{'train'};
			}
			if($foo->{'test'} > $max_test_mse)
			{
				$max_test_mse = $foo->{'test'};
			}
			if($foo->{'train'} > $max_train_mse)
			{
				$max_train_mse = $foo->{'train'};
			}
		}
	
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
				}
				
				if(!exists($foo->{'train_rank'}) && (!$min_algo_train || $foo->{'train'} < $summary{$min_algo_train}->{'train'}))
				{
					$min_algo_train = $algo;
				}
			}
	
			$summary{$min_algo_test}->{'test_rank'} = $i;
			$summary{$min_algo_train}->{'train_rank'} = $i;
			$summary{$min_algo_test}->{'test_percent'} = (($summary{$min_algo_test}->{'test'}-$min_test_mse)*100)/($max_test_mse-$min_test_mse);
			$summary{$min_algo_train}->{'train_percent'} = (($summary{$min_algo_train}->{'train'}-$min_train_mse)*100)/($max_train_mse-$min_train_mse);
		}
	
		#write tex to file
		open(OUT, "> $path".$oldDataSet."_table.tex") or die("unable to open file $path".$oldDataSet."_table.tex");
		
		print OUT '\begin{tabular} {|l|r|r|r|r|r|r|}'."\n";
		print OUT '\hline'."\n";
		print OUT '& \multicolumn{3}{c|}{\textbf{Best Train}} & \multicolumn{3}{c|}{\textbf{Best Test}} \\\\'."\n";
		print OUT '\hline'."\n";
		print OUT '\textbf{Configuration} & \textbf{MSE} & \textbf{Rank} & \textbf{\%} & \textbf{MSE} & \textbf{Rank} & \textbf{\%} \\\\'."\n";
		print OUT '\hline'."\n";
	
		foreach $algo (@algorithms)
		{ 
			$foo = $summary{$algo->{ALGO}};

			$highlight_train_start = "";
			$highlight_train_end = "";
			$highlight_test_start = "";
			$highlight_test_end = "";
			
			if($foo->{'train'} == $min_train_mse)
			{
				$highlight_train_start = "\\textcolor{green}{";
				$highlight_train_end = "}";
			}
			elsif($foo->{'train'} == $max_train_mse)
			{
				$highlight_train_start = "\\textcolor{red}{";
				$highlight_train_end = "}";
			}
			
			if($foo->{'test'} == $min_test_mse)
			{
				$highlight_test_start = "\\textcolor{green}{";
				$highlight_test_end = "}";
			}
			elsif($foo->{'test'} == $max_test_mse)
			{
				$highlight_test_start = "\\textcolor{red}{";
				$highlight_test_end = "}";
			}
			
			printf OUT ("%s & $highlight_train_start %.8f $highlight_train_end & $highlight_train_start %d $highlight_train_end & $highlight_train_start %.2f $highlight_train_end & $highlight_test_start %.8f $highlight_test_end & $highlight_test_start %d $highlight_test_end & $highlight_test_start %.2f $highlight_test_end \\\\\n\\hline\n", 
				$algo->{NAME}, 
				$foo->{'train'}, $foo->{'train_rank'}, $foo->{'train_percent'}, 
				$foo->{'test'}, $foo->{'test_rank'}, $foo->{'test_percent'});
	
			$total{$algo->{ALGO}}->{'train'} += $foo->{'train'};
			$total{$algo->{ALGO}}->{'test'} += $foo->{'test'};
			$total{$algo->{ALGO}}->{'train_percent'} += $foo->{'train_percent'};
			$total{$algo->{ALGO}}->{'test_percent'} += $foo->{'test_percent'};
			$total{$algo->{ALGO}}->{'train_rank'} += $foo->{'train_rank'};
			$total{$algo->{ALGO}}->{'test_rank'} += $foo->{'test_rank'};
		}
		print OUT '\end{tabular}'."\n";
		close(OUT);
	
		undef(%summary);
	}
	
	if($found_algo) #only the algorithms in the array
	{
		$summary{$algorithm}->{$test_train} = $mse;
	}
	$oldDataSet = $dataSet;
}

print "**Average**\n";
open(OUT, "> ".$path."average_table.tex") or die("unable to open file ".$path."average_table.tex");

#get min and max in order to make colors in table
$foo = $total{$algorithms[0]->{ALGO}};
$min_train = $max_train = $foo->{'train'};
$min_train_rank = $max_train_rank = $foo->{'train_rank'};
$min_train_percent = $max_train_percent = $foo->{'train_percent'};
$min_test = $max_test = $foo->{'test'};
$min_test_rank = $max_test_rank = $foo->{'test_rank'};
$min_test_percent = $max_test_percent = $foo->{'test_percent'};

foreach $algo (@algorithms)
{
    $foo = $total{$algo->{ALGO}};
    
    #train
    if($foo->{'train'} < $min_train)
    {
    	$min_train = $foo->{'train'};
    }
    elsif($foo->{'train'} > $max_train)
    {
    	$max_train = $foo->{'train'};
    }
    
    if($foo->{'train_rank'} < $min_train_rank)
    {
    	$min_train_rank = $foo->{'train_rank'};
    }
    elsif($foo->{'train_rank'} > $max_train_rank)
    {
    	$max_train_rank = $foo->{'train_rank'};
    }
    
    if($foo->{'train_percent'} < $min_train_percent)
    {
    	$min_train_percent = $foo->{'train_percent'};
    }
    elsif($foo->{'train_percent'} > $max_train_percent)
    {
    	$max_train_percent = $foo->{'train_percent'};
    }

	#test
    if($foo->{'test'} < $min_test)
    {
    	$min_test = $foo->{'test'};
    }
    elsif($foo->{'test'} > $max_test)
    {
    	$max_test = $foo->{'test'};
    }
    
    if($foo->{'test_rank'} < $min_test_rank)
    {
    	$min_test_rank = $foo->{'test_rank'};
    }
    elsif($foo->{'test_rank'} > $max_test_rank)
    {
    	$max_test_rank = $foo->{'test_rank'};
    }
    
    if($foo->{'test_percent'} < $min_test_percent)
    {
    	$min_test_percent = $foo->{'test_percent'};
    }
    elsif($foo->{'test_percent'} > $max_test_percent)
    {
    	$max_test_percent = $foo->{'test_percent'};
    }
}

print OUT '\begin{tabular} {|l|r|r|r|r|r|r|}'."\n";
print OUT '\hline'."\n";
print OUT '& \multicolumn{3}{c|}{\textbf{Best Train}} & \multicolumn{3}{c|}{\textbf{Best Test}} \\\\'."\n";
print OUT '\hline'."\n";
print OUT '\textbf{Configuration} & \textbf{MSE} & \textbf{Rank} & \textbf{\%} & \textbf{MSE} & \textbf{Rank} & \textbf{\%} \\\\'."\n";
print OUT '\hline'."\n";
foreach $algo (@algorithms)
{
    $foo = $total{$algo->{ALGO}};
    
	$highlight_train_start = "";
	$highlight_train_end = "";
	$highlight_train_rank_start = "";
	$highlight_train_rank_end = "";
	$highlight_train_percent_start = "";
	$highlight_train_percent_end = "";
    
	if($foo->{'train'} == $min_train)
	{
		$highlight_train_start = "\\textcolor{green}{";
		$highlight_train_end = "}";
	}
	elsif($foo->{'train'} == $max_train)
	{
		$highlight_train_start = "\\textcolor{red}{";
		$highlight_train_end = "}";
	}
			
	if($foo->{'train_rank'} == $min_train_rank)
	{
		$highlight_train_rank_start = "\\textcolor{green}{";
		$highlight_train_rank_end = "}";
	}
	elsif($foo->{'train_rank'} == $max_train_rank)
	{
		$highlight_train_rank_start = "\\textcolor{red}{";
		$highlight_train_rank_end = "}";
	}
			
	if($foo->{'train_percent'} == $min_train_percent)
	{
		$highlight_train_percent_start = "\\textcolor{green}{";
		$highlight_train_percent_end = "}";
	}
	elsif($foo->{'train_percent'} == $max_train_percent)
	{
		$highlight_train_percent_start = "\\textcolor{red}{";
		$highlight_train_percent_end = "}";
	}
			
	$highlight_test_start = "";
	$highlight_test_end = "";
	$highlight_test_rank_start = "";
	$highlight_test_rank_end = "";
	$highlight_test_percent_start = "";
	$highlight_test_percent_end = "";
    
	if($foo->{'test'} == $min_test)
	{
		$highlight_test_start = "\\textcolor{green}{";
		$highlight_test_end = "}";
	}
	elsif($foo->{'test'} == $max_test)
	{
		$highlight_test_start = "\\textcolor{red}{";
		$highlight_test_end = "}";
	}
			
	if($foo->{'test_rank'} == $min_test_rank)
	{
		$highlight_test_rank_start = "\\textcolor{green}{";
		$highlight_test_rank_end = "}";
	}
	elsif($foo->{'test_rank'} == $max_test_rank)
	{
		$highlight_test_rank_start = "\\textcolor{red}{";
		$highlight_test_rank_end = "}";
	}
			
	if($foo->{'test_percent'} == $min_test_percent)
	{
		$highlight_test_percent_start = "\\textcolor{green}{";
		$highlight_test_percent_end = "}";
	}
	elsif($foo->{'test_percent'} == $max_test_percent)
	{
		$highlight_test_percent_start = "\\textcolor{red}{";
		$highlight_test_percent_end = "}";
	}
    
    printf OUT ("%s & $highlight_train_start %.6f $highlight_train_end & $highlight_train_rank_start %.2f $highlight_train_rank_end & $highlight_train_percent_start %.2f $highlight_train_percent_end & $highlight_test_start %.6f $highlight_test_end & $highlight_test_rank_start %.2f $highlight_test_rank_end & $highlight_test_percent_start %.2f $highlight_test_percent_end \\\\\n\\hline\n", 
	   $algo->{NAME}, 
	   $foo->{'train'}/$numDataSet, $foo->{'train_rank'}/$numDataSet, $foo->{'train_percent'}/$numDataSet, 
	   $foo->{'test'}/$numDataSet, $foo->{'test_rank'}/$numDataSet, $foo->{'test_percent'}/$numDataSet);
}
print OUT '\end{tabular}'."\n";
close(OUT);
