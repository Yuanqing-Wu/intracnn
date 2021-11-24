python test.py \
--cu_size=64x64 \
--data_dir=/home/wgq/research/bs/dataset/allintra \
--test_list=test_list_64x64.txt \
--model=model/64x64/model_epoch7000.pth \
> model/64x64/test7000.log &

python test.py \
--cu_size=32x32 \
--data_dir=/home/wgq/research/bs/dataset/allintra \
--test_list=test_list_32x32.txt \
--model=model/32x32/model_epoch6500.pth \
> model/32x32/test6500.log &

python test.py \
--cu_size=32x32 \
--data_dir=/home/wgq/research/bs/dataset/allintra \
--test_list=test_list_32x32.txt \
--model=model/32x32/model_epoch800.pth \
> model/32x32/test800.log &

python test.py \
--cu_size=32x32 \
--data_dir=/home/wgq/research/bs/dataset/allintra \
--test_list=test_list_32x32.txt \
--model=model/32x32/model_epoch1000.pth \
> model/32x32/test1000.log &

python test.py \
--cu_size=32x16 \
--data_dir=/home/wgq/research/bs/dataset/allintra \
--test_list=test_list_32x16.txt \
--model=model/32x16/model_epoch6000.pth \
> model/32x16/test600.log &

python test.py \
--cu_size=32x8 \
--data_dir=/home/wgq/research/bs/dataset/allintra \
--test_list=test_list_32x8.txt \
--model=model/32x8/model_epoch6000.pth \
> model/32x8/test6000.log &

python test.py \
--cu_size=16x16 \
--data_dir=/home/wgq/research/bs/dataset/allintra \
--test_list=test_list_16x16.txt \
--model=model/16x16/model_epoch6000.pth \
> model/16x16/test6000.log &

python test.py \
--cu_size=16x8 \
--data_dir=/home/wgq/research/bs/dataset/allintra \
--test_list=test_list_16x8.txt \
--model=model/16x8/model_epoch6000.pth \
> model/16x8/test6000.log &

python test.py \
--cu_size=8x8 \
--data_dir=/home/wgq/research/bs/dataset/allintra \
--test_list=test_list_8x8.txt \
--model=model/8x8/model_epoch6000.pth \
> model/8x8/test6000.log &
