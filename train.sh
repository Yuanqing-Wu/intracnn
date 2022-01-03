python train.py \
--cu_size=64x64 \
--data_dir=/home/wgq/research/bs/dataset/allintra \
--train_list=ttrain_list_64x64.txt \
--test_list=ttest_list_64x64.txt \
--output=model/64x64 \
--model=model/64x64/model_epoch6000.pth \
--pretrain_epoch=6000 \
> model/64x64/train6000.log &

python train.py \
--cu_size=32x32 \
--data_dir=/home/wgq/research/bs/dataset/allintra \
--train_list=ttrain_list_32x32.txt \
--test_list=ttest_list_32x32.txt \
--output=model/32x32 \
--model=model/32x32/model_epoch2500.pth \
--pretrain_epoch=2500 \
> model/32x32/train2500.log &

python train.py \
--cu_size=32x16 \
--data_dir=/home/wgq/research/bs/dataset/allintra \
--train_list=ttrain_list_32x16.txt \
--test_list=ttest_list_32x16.txt \
--output=model/32x16 \
--model=model/32x16/model_epoch2400.pth \
--pretrain_epoch=2400 \
> model/32x16/train2400.log &

python train.py \
--cu_size=32x8 \
--data_dir=/home/wgq/research/bs/dataset/allintra \
--train_list=ttrain_list_32x8.txt \
--test_list=ttest_list_32x8.txt \
--output=model/32x8 \
--model=model/32x8/model_epoch2400.pth \
--pretrain_epoch=2400 \
> model/32x8/train2400.log &

python train.py \
--cu_size=16x16 \
--data_dir=/home/wgq/research/bs/dataset/allintra \
--train_list=ttrain_list_16x16.txt \
--test_list=ttest_list_16x16.txt \
--output=model/16x16 \
--model=model/16x16/model_epoch2400.pth \
--pretrain_epoch=2400 \
> model/16x16/train2400.log &

python train.py \
--cu_size=16x8 \
--data_dir=/home/wgq/research/bs/dataset/allintra \
--train_list=ttrain_list_16x8.txt \
--test_list=ttest_list_16x8.txt \
--output=model/16x8 \
--model=model/16x8/model_epoch2900.pth \
--pretrain_epoch=2900 \
> model/16x8/train2900.log &

python train.py \
--cu_size=8x8 \
--data_dir=/home/wgq/research/bs/dataset/allintra \
--train_list=ttrain_list_8x8.txt \
--test_list=ttest_list_8x8.txt \
--output=model/8x8 \
--model=model/8x8/model_epoch2600.pth \
--pretrain_epoch=2600 \
> model/8x8/train2600.log &