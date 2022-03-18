export seed=1
export NGPUS=4
export base_dir=.
export dataset=_gta5
export attr=_bin_small
export DIS=binary
export weight=https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth


# warm-up
python -m torch.distributed.launch --nproc_per_node=$NGPUS train.py -cfg configs/warm_up$dataset.yaml data_dir $base_dir/datasets OUTPUT_DIR $base_dir/results/warm_up$dataset$attr$seed MODEL.WEIGHTS $weight SOLVER.DIS $DIS  warm_up True  > $base_dir/logs/warm_up$dataset$attr$seed.log 2>&1


python test.py -cfg configs/warm_up$dataset.yaml --saveres data_dir $base_dir/datasets OUTPUT_DIR $base_dir/datasets/cityscapes/warm_up$dataset$attr$seed resume $base_dir/pth/warm_up$dataset$attr$seed/model_last.pth  DATASETS.TEST cityscapes_train  MODEL.WEIGHTS $weight SOLVER.DIS $DIS  > $base_dir/logs/ps_warm_up$dataset$attr$seed.log 2>&1

# train for 3 rounds
python -m torch.distributed.launch --nproc_per_node=$NGPUS train.py -cfg configs/train$dataset.yaml data_dir $base_dir/datasets OUTPUT_DIR $base_dir/results/train1$dataset$attr$seed distill_path cityscapes/warm_up$dataset$attr$seed/inference/cityscapes_train MODEL.WEIGHTS $weight SOLVER.DIS $DIS > $base_dir/logs/train1$dataset$attr$seed.log 2>&1

python test.py -cfg configs/train$dataset.yaml --saveres data_dir $base_dir/datasets OUTPUT_DIR $base_dir/datasets/cityscapes/train1$dataset$attr$seed resume $base_dir/pth/train1$dataset$attr$seed/model_last.pth  DATASETS.TEST cityscapes_train  MODEL.WEIGHTS $weight SOLVER.DIS $DIS  > $base_dir/logs/ps_train1$dataset$attr$seed.log 2>&1

python -m torch.distributed.launch --nproc_per_node=$NGPUS train.py -cfg configs/train$dataset.yaml data_dir $base_dir/datasets OUTPUT_DIR $base_dir/results/train2$dataset$attr$seed distill_path cityscapes/train1$dataset$attr$seed/inference/cityscapes_train MODEL.WEIGHTS $weight SOLVER.DIS $DIS > $base_dir/logs/train2$dataset$attr$seed.log 2>&1

python test.py -cfg configs/train$dataset.yaml --saveres data_dir $base_dir/datasets OUTPUT_DIR $base_dir/datasets/cityscapes/train2$dataset$attr$seed resume $base_dir/pth/train2$dataset$attr$seed/model_last.pth  DATASETS.TEST cityscapes_train  MODEL.WEIGHTS $weight SOLVER.DIS $DIS  > $base_dir/logs/ps_train2$dataset$attr$seed.log 2>&1

python -m torch.distributed.launch --nproc_per_node=$NGPUS train.py -cfg configs/train$dataset.yaml data_dir $base_dir/datasets OUTPUT_DIR $base_dir/results/train3$dataset$attr$seed distill_path cityscapes/train2$dataset$attr$seed/inference/cityscapes_train  MODEL.WEIGHTS $weight SOLVER.DIS $DIS > $base_dir/logs/train3$dataset$attr$seed.log 2>&1

#Comment any of the following 
# validate
python test.py -cfg configs/train$dataset.yaml --saveres data_dir $base_dir/datasets OUTPUT_DIR $base_dir/datasets/cityscapes/train3$dataset$attr$seed resume $base_dir/pth/train3$dataset$attr$seed/model_last.pth MODEL.WEIGHTS $weight SOLVER.DIS $DIS  > $base_dir/logs/validation$dataset$attr$seed.log 2>&1
# test for submit
python test.py -cfg configs/train$dataset.yaml data_dir $base_dir/datasets OUTPUT_DIR $base_dir/datasets/cityscapes/train3$dataset$attr$seed resume $base_dir/pth/train3$dataset$attr$seed/model_last.pth  DATASETS.TEST cityscapes_test  MODEL.WEIGHTS $weight SOLVER.DIS $DIS  > $base_dir/logs/test$dataset$attr$seed.log 2>&1
