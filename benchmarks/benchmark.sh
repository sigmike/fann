#!/bin/sh

seconds_training=1000;
secs_between_reports=1;

function benchmark_algorithm() {
    ./quality $algo datasets/$prob.train datasets/$prob.test $prob.$algo.train.out $prob.$algo.test.out $n1 $n2 $seconds_training $secs_between_reports
}

function benchmark_problem() {
    rm -f *_fixed.net
    algo="fann_rprop"; benchmark_algorithm;
    ./quality_fixed $prob.$algo.train.out_fixed_train $prob.$algo.train.out_fixed_test $prob.$algo.fixed_train.out $prob.$algo.fixed_test.out *_fixed.net
    algo="fann_rprop_stepwise"; benchmark_algorithm;
    algo="fann_quickprop"; benchmark_algorithm;
    #algo="fann_quickprop_stepwise"; benchmark_algorithm;
    algo="fann_batch"; benchmark_algorithm;
    #algo="fann_batch_stepwise"; benchmark_algorithm;
    algo="fann_incremental"; benchmark_algorithm;
    #algo="fann_incremental_stepwise"; benchmark_algorithm;

    #comment out two following lines if the libraries are not available
    algo="lwnn"; benchmark_algorithm;
    algo="jneural"; benchmark_algorithm;
}

#comment out some of the lines below if some of the problems should not be benchmarked

prob="building"; n1=16; n2=0;
benchmark_problem;
exit 1;
prob="cancer"; n1=8; n2=4;
benchmark_problem;

prob="card"; n1=32; n2=0;
benchmark_problem;

prob="diabetes"; n1=4; n2=0;
benchmark_problem;

prob="flare"; n1=4; n2=0;
benchmark_problem;

prob="gene"; n1=4; n2=2;
benchmark_problem;

prob="glass"; n1=32; n2=0;
benchmark_problem;

prob="heart"; n1=16; n2=8;
benchmark_problem;

prob="horse"; n1=4; n2=4;
benchmark_problem;

prob="mushroom"; n1=32; n2=0;
benchmark_problem;

prob="robot"; n1=96; n2=0;
benchmark_problem;

prob="soybean"; n1=16; n2=8;
benchmark_problem;

prob="thyroid"; n1=16; n2=8;
benchmark_problem;

./performance fann fann_performance.out 1 2048 2 20
./performance fann_stepwise fann_stepwise_performance.out 1 2048 2 20
./performance_fixed fann fann_fixed_performance.out 1 2048 2 20
./performance lwnn lwnn_performance.out 1 2048 2 20
./performance jneural jneural_performance.out 1 256 2 20

