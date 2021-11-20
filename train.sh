python train.py \
--cu_size=64x64 \
--data_dir=/home/wgq/research/bs/dataset/allintra \
--train_list=train_list_64x64.txt \
--test_list=test_list_64x64.txt \
--output=model/64x64 \
> model/64x64/train.log &

python train.py \
--cu_size=32x32 \
--data_dir=/home/wgq/research/bs/dataset/allintra \
--train_list=train_list_32x32.txt \
--test_list=test_list_32x32.txt \
--output=model/32x32 \
> model/32x32/train.log &

python train.py \
--cu_size=32x16 \
--data_dir=/home/wgq/research/bs/dataset/allintra \
--train_list=train_list_32x16.txt \
--test_list=test_list_32x16.txt \
--output=model/32x16 \
> model/32x16/train.log &

python train.py \
--cu_size=32x8 \
--data_dir=/home/wgq/research/bs/dataset/allintra \
--train_list=train_list_32x8.txt \
--test_list=test_list_32x8.txt \
--output=model/32x8 \
> model/32x8/train.log &

python train.py \
--cu_size=16x16 \
--data_dir=/home/wgq/research/bs/dataset/allintra \
--train_list=train_list_16x16.txt \
--test_list=test_list_16x16.txt \
--output=model/16x16 \
> model/16x16/train.log &

python train.py \
--cu_size=16x8 \
--data_dir=/home/wgq/research/bs/dataset/allintra \
--train_list=train_list_16x8.txt \
--test_list=test_list_16x8.txt \
--output=model/16x8 \
> model/16x8/train.log &

python train.py \
--cu_size=8x8 \
--data_dir=/home/wgq/research/bs/dataset/allintra \
--train_list=train_list_8x8.txt \
--test_list=test_list_8x8.txt \
--output=model/8x8 \
> model/8x8/train.log &