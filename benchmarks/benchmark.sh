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

rm -f *_fixed.net
./quality fann datasets/building.train datasets/building.test building_fann_train.out building_fann_test.out 16 0 200 1
./quality_fixed building_fann_train.out_fixed_train building_fann_train.out_fixed_test building_fann_fixed_train.out building_fann_fixed_test.out *_fixed.net
./quality fann_stepwise datasets/building.train datasets/building.test building_fann_stepwise_train.out building_fann_stepwise_test.out 16 0 200 1
./quality lwnn datasets/building.train datasets/building.test building_lwnn_train.out building_lwnn_test.out 16 0 200 1
./quality jneural datasets/building.train datasets/building.test building_jneural_train.out building_jneural_test.out 16 0 200 1

rm -f *_fixed.net
./quality fann datasets/cancer.train datasets/cancer.test cancer_fann_train.out cancer_fann_test.out 8 4 200 1
./quality_fixed cancer_fann_train.out_fixed_train cancer_fann_train.out_fixed_test cancer_fann_fixed_train.out cancer_fann_fixed_test.out *_fixed.net
./quality fann_stepwise datasets/cancer.train datasets/cancer.test cancer_fann_stepwise_train.out cancer_fann_stepwise_test.out 8 4 200 1
./quality lwnn datasets/cancer.train datasets/cancer.test cancer_lwnn_train.out cancer_lwnn_test.out 8 4 200 1
./quality jneural datasets/cancer.train datasets/cancer.test cancer_jneural_train.out cancer_jneural_test.out 8 4 200 1

rm -f *_fixed.net
./quality fann datasets/card.train datasets/card.test card_fann_train.out card_fann_test.out 32 0 200 1
./quality_fixed card_fann_train.out_fixed_train card_fann_train.out_fixed_test card_fann_fixed_train.out card_fann_fixed_test.out *_fixed.net
./quality fann_stepwise datasets/card.train datasets/card.test card_fann_stepwise_train.out card_fann_stepwise_test.out 32 0 200 1
./quality lwnn datasets/card.train datasets/card.test card_lwnn_train.out card_lwnn_test.out 32 0 200 1
./quality jneural datasets/card.train datasets/card.test card_jneural_train.out card_jneural_test.out 32 0 200 1

rm -f *_fixed.net
./quality fann datasets/diabetes.train datasets/diabetes.test diabetes_fann_train.out diabetes_fann_test.out 8 8 200 1
./quality_fixed diabetes_fann_train.out_fixed_train diabetes_fann_train.out_fixed_test diabetes_fann_fixed_train.out diabetes_fann_fixed_test.out *_fixed.net
./quality fann_stepwise datasets/diabetes.train datasets/diabetes.test diabetes_fann_stepwise_train.out diabetes_fann_stepwise_test.out 8 8 200 1
./quality lwnn datasets/diabetes.train datasets/diabetes.test diabetes_lwnn_train.out diabetes_lwnn_test.out 8 8 200 1
./quality jneural datasets/diabetes.train datasets/diabetes.test diabetes_jneural_train.out diabetes_jneural_test.out 8 8 200 1

rm -f *_fixed.net
./quality fann datasets/flare.train datasets/flare.test flare_fann_train.out flare_fann_test.out 4 0 200 1
./quality_fixed flare_fann_train.out_fixed_train flare_fann_train.out_fixed_test flare_fann_fixed_train.out flare_fann_fixed_test.out *_fixed.net
./quality fann_stepwise datasets/flare.train datasets/flare.test flare_fann_stepwise_train.out flare_fann_stepwise_test.out 4 0 200 1
./quality lwnn datasets/flare.train datasets/flare.test flare_lwnn_train.out flare_lwnn_test.out 4 0 200 1
./quality jneural datasets/flare.train datasets/flare.test flare_jneural_train.out flare_jneural_test.out 4 0 200 1

rm -f *_fixed.net
./quality fann datasets/gene.train datasets/gene.test gene_fann_train.out gene_fann_test.out 4 2 200 1
./quality_fixed gene_fann_train.out_fixed_train gene_fann_train.out_fixed_test gene_fann_fixed_train.out gene_fann_fixed_test.out *_fixed.net
./quality fann_stepwise datasets/gene.train datasets/gene.test gene_fann_stepwise_train.out gene_fann_stepwise_test.out 4 2 200 1
./quality lwnn datasets/gene.train datasets/gene.test gene_lwnn_train.out gene_lwnn_test.out 4 2 200 1
./quality jneural datasets/gene.train datasets/gene.test gene_jneural_train.out gene_jneural_test.out 4 2 200 1

rm -f *_fixed.net
./quality fann datasets/glass.train datasets/glass.test glass_fann_train.out glass_fann_test.out 32 0 200 1
./quality_fixed glass_fann_train.out_fixed_train glass_fann_train.out_fixed_test glass_fann_fixed_train.out glass_fann_fixed_test.out *_fixed.net
./quality fann_stepwise datasets/glass.train datasets/glass.test glass_fann_stepwise_train.out glass_fann_stepwise_test.out 32 0 200 1
./quality lwnn datasets/glass.train datasets/glass.test glass_lwnn_train.out glass_lwnn_test.out 32 0 200 1
./quality jneural datasets/glass.train datasets/glass.test glass_jneural_train.out glass_jneural_test.out 32 0 200 1

rm -f *_fixed.net
./quality fann datasets/heart.train datasets/heart.test heart_fann_train.out heart_fann_test.out 16 8 200 1
./quality_fixed heart_fann_train.out_fixed_train heart_fann_train.out_fixed_test heart_fann_fixed_train.out heart_fann_fixed_test.out *_fixed.net
./quality fann_stepwise datasets/heart.train datasets/heart.test heart_fann_stepwise_train.out heart_fann_stepwise_test.out 16 8 200 1
./quality lwnn datasets/heart.train datasets/heart.test heart_lwnn_train.out heart_lwnn_test.out 16 8 200 1
./quality jneural datasets/heart.train datasets/heart.test heart_jneural_train.out heart_jneural_test.out 16 8 200 1

rm -f *_fixed.net
./quality fann datasets/horse.train datasets/horse.test horse_fann_train.out horse_fann_test.out 4 4 200 1
./quality_fixed horse_fann_train.out_fixed_train horse_fann_train.out_fixed_test horse_fann_fixed_train.out horse_fann_fixed_test.out *_fixed.net
./quality fann_stepwise datasets/horse.train datasets/horse.test horse_fann_stepwise_train.out horse_fann_stepwise_test.out 4 4 200 1
./quality lwnn datasets/horse.train datasets/horse.test horse_lwnn_train.out horse_lwnn_test.out 4 4 200 1
./quality jneural datasets/horse.train datasets/horse.test horse_jneural_train.out horse_jneural_test.out 4 4 200 1

rm -f *_fixed.net
./quality fann datasets/mushroom.train datasets/mushroom.test mushroom_fann_train.out mushroom_fann_test.out 32 0 200 1
./quality_fixed mushroom_fann_train.out_fixed_train mushroom_fann_train.out_fixed_test mushroom_fann_fixed_train.out mushroom_fann_fixed_test.out *_fixed.net
./quality fann_stepwise datasets/mushroom.train datasets/mushroom.test mushroom_fann_stepwise_train.out mushroom_fann_stepwise_test.out 32 0 200 1
./quality lwnn datasets/mushroom.train datasets/mushroom.test mushroom_lwnn_train.out mushroom_lwnn_test.out 32 0 200 1
./quality jneural datasets/mushroom.train datasets/mushroom.test mushroom_jneural_train.out mushroom_jneural_test.out 32 0 200 1

rm -f *_fixed.net
./quality fann datasets/robot.train datasets/robot.test robot_fann_train.out robot_fann_test.out 96 0 200 1
./quality_fixed robot_fann_train.out_fixed_train robot_fann_train.out_fixed_test robot_fann_fixed_train.out robot_fann_fixed_test.out *_fixed.net
./quality fann_stepwise datasets/robot.train datasets/robot.test robot_fann_stepwise_train.out robot_fann_stepwise_test.out 96 0 200 1
./quality lwnn datasets/robot.train datasets/robot.test robot_lwnn_train.out robot_lwnn_test.out 96 0 200 1
./quality jneural datasets/robot.train datasets/robot.test robot_jneural_train.out robot_jneural_test.out 96 0 200 1

rm -f *_fixed.net
./quality fann datasets/soybean.train datasets/soybean.test soybean_fann_train.out soybean_fann_test.out 16 8 200 1
./quality_fixed soybean_fann_train.out_fixed_train soybean_fann_train.out_fixed_test soybean_fann_fixed_train.out soybean_fann_fixed_test.out *_fixed.net
./quality fann_stepwise datasets/soybean.train datasets/soybean.test soybean_fann_stepwise_train.out soybean_fann_stepwise_test.out 16 8 200 1
./quality lwnn datasets/soybean.train datasets/soybean.test soybean_lwnn_train.out soybean_lwnn_test.out 16 8 200 1
./quality jneural datasets/soybean.train datasets/soybean.test soybean_jneural_train.out soybean_jneural_test.out 16 8 200 1

rm -f *_fixed.net
./quality fann datasets/thyroid.train datasets/thyroid.test thyroid_fann_train.out thyroid_fann_test.out 16 8 200 1
./quality_fixed thyroid_fann_train.out_fixed_train thyroid_fann_train.out_fixed_test thyroid_fann_fixed_train.out thyroid_fann_fixed_test.out *_fixed.net
./quality fann_stepwise datasets/thyroid.train datasets/thyroid.test thyroid_fann_stepwise_train.out thyroid_fann_stepwise_test.out 16 8 200 1
./quality lwnn datasets/thyroid.train datasets/thyroid.test thyroid_lwnn_train.out thyroid_lwnn_test.out 16 8 200 1
./quality jneural datasets/thyroid.train datasets/thyroid.test thyroid_jneural_train.out thyroid_jneural_test.out 16 8 200 1

