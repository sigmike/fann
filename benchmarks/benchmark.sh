#!/bin/sh
./performance fann fann_performance.out 1 2048 2 20
./performance fann_stepwise fann_stepwise_performance.out 1 2048 2 20
./performance_fixed fann fann_fixed_performance.out 1 2048 2 20
./performance lwnn lwnn_performance.out 1 2048 2 20
./performance jneural jneural_performance.out 1 512 2 20

#./performance_arm fann fann_performance_arm.out 1 512 2 20
#./performance_arm fann_noopt fann_noopt_performance_arm.out 1 512 2 20
#./performance_arm fann_thres fann_thres_performance_arm.out 1 512 2 20
#./performance_fixed_arm fann fann_fixed_performance_arm.out 1 512 2 20
#./performance_arm lwnn lwnn_performance_arm.out 1 512 2 20
#./performance_arm jneural jneural_performance_arm.out 1 512 2 20

./quality fann datasets/mydata/building.train datasets/mydata/building.test building_fann_train.out building_fann_test.out 16 0 200 1
./quality_fixed building_fann_train.out_fixed_train building_fann_train.out_fixed_test building_fann_fixed_train.out building_fann_fixed_test.out *_fixed
./quality fann_stepwise datasets/mydata/building.train datasets/mydata/building.test building_fann_stepwise_train.out building_fann_stepwise_test.out 16 0 200 1
./quality lwnn datasets/mydata/building.train datasets/mydata/building.test building_lwnn_train.out building_lwnn_test.out 16 0 200 1
./quality jneural datasets/mydata/building.train datasets/mydata/building.test building_jneural_train.out building_jneural_test.out 16 0 200 1

./quality fann datasets/mydata/card.train datasets/mydata/card.test card_fann_train.out card_fann_test.out 32 0 200 1
./quality_fixed card_fann_train.out_fixed_train card_fann_train.out_fixed_test card_fann_fixed_train.out card_fann_fixed_test.out *_fixed
./quality fann_stepwise datasets/mydata/card.train datasets/mydata/card.test card_fann_stepwise_train.out card_fann_stepwise_test.out 32 0 200 1
./quality lwnn datasets/mydata/card.train datasets/mydata/card.test card_lwnn_train.out card_lwnn_test.out 32 0 200 1
./quality jneural datasets/mydata/card.train datasets/mydata/card.test card_jneural_train.out card_jneural_test.out 32 0 200 1

./quality fann datasets/mydata/gene.train datasets/mydata/gene.test gene_fann_train.out gene_fann_test.out 4 2 200 1
./quality_fixed gene_fann_train.out_fixed_train gene_fann_train.out_fixed_test gene_fann_fixed_train.out gene_fann_fixed_test.out *_fixed
./quality fann_stepwise datasets/mydata/gene.train datasets/mydata/gene.test gene_fann_stepwise_train.out gene_fann_stepwise_test.out 4 2 200 1
./quality lwnn datasets/mydata/gene.train datasets/mydata/gene.test gene_lwnn_train.out gene_lwnn_test.out 4 2 200 1
./quality jneural datasets/mydata/gene.train datasets/mydata/gene.test gene_jneural_train.out gene_jneural_test.out 4 2 200 1

./quality fann datasets/mydata/mushroom.train datasets/mydata/mushroom.test mushroom_fann_train.out mushroom_fann_test.out 32 0 200 1
./quality_fixed mushroom_fann_train.out_fixed_train mushroom_fann_train.out_fixed_test mushroom_fann_fixed_train.out mushroom_fann_fixed_test.out *_fixed
./quality fann_stepwise datasets/mydata/mushroom.train datasets/mydata/mushroom.test mushroom_fann_stepwise_train.out mushroom_fann_stepwise_test.out 32 0 200 1
./quality lwnn datasets/mydata/mushroom.train datasets/mydata/mushroom.test mushroom_lwnn_train.out mushroom_lwnn_test.out 32 0 200 1
./quality jneural datasets/mydata/mushroom.train datasets/mydata/mushroom.test mushroom_jneural_train.out mushroom_jneural_test.out 32 0 200 1

./quality fann datasets/mydata/soybean.train datasets/mydata/soybean.test soybean_fann_train.out soybean_fann_test.out 16 8 200 1
./quality_fixed soybean_fann_train.out_fixed_train soybean_fann_train.out_fixed_test soybean_fann_fixed_train.out soybean_fann_fixed_test.out *_fixed
./quality fann_stepwise datasets/mydata/soybean.train datasets/mydata/soybean.test soybean_fann_stepwise_train.out soybean_fann_stepwise_test.out 16 8 200 1
./quality lwnn datasets/mydata/soybean.train datasets/mydata/soybean.test soybean_lwnn_train.out soybean_lwnn_test.out 16 8 200 1
./quality jneural datasets/mydata/soybean.train datasets/mydata/soybean.test soybean_jneural_train.out soybean_jneural_test.out 16 8 200 1

./quality fann datasets/mydata/thyroid.train datasets/mydata/thyroid.test thyroid_fann_train.out thyroid_fann_test.out 16 8 200 1
./quality_fixed thyroid_fann_train.out_fixed_train thyroid_fann_train.out_fixed_test thyroid_fann_fixed_train.out thyroid_fann_fixed_test.out *_fixed
./quality fann_stepwise datasets/mydata/thyroid.train datasets/mydata/thyroid.test thyroid_fann_stepwise_train.out thyroid_fann_stepwise_test.out 16 8 200 1
./quality lwnn datasets/mydata/thyroid.train datasets/mydata/thyroid.test thyroid_lwnn_train.out thyroid_lwnn_test.out 16 8 200 1
./quality jneural datasets/mydata/thyroid.train datasets/mydata/thyroid.test thyroid_jneural_train.out thyroid_jneural_test.out 16 8 200 1
