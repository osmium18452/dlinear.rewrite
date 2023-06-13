#!/bin/sh
# model: informer, autoformer, fedformer, pyraformer, transformer, crossformer, reformer, linear, dlinear, nlinear
# dataset: wht, gweather, etth1, etth2, ettm1, ettm2, exchange, ill
# hyper parameters:
# G: gpu
# M: multi gpu
# D: delete model
# B: best model //with or without
# S: save path
# s: stride
# o: output length 24, 48, 96, 192, 360, 720
#    or 20, 40, 80, 160, 320
#
epoch=3
mem_scalar=1
cuda_device=2,3,4,5,6,7
dev_num=6

for model in informer autoformer fedformer pyraformer transformer reformer; do
  for dataset in gweather etth1 etth2 ettm1 ettm2 exchange ill; do
    for output_len in 24 48 96 192 360 720; do
      if [ $model = "reformer" ]; then
        batch_size=$(expr 30 \* $mem_scalar)
      elif [ $model = 'informer' ]; then
        batch_size=$(expr 40 \* $mem_scalar)
      elif [ $model = 'autoformer' ]; then
        batch_size=$(expr 25 \* $mem_scalar)
      elif [ $model = 'fedformer' ]; then
        batch_size=$(expr 90 \* $mem_scalar)
      elif [ $model = 'pyraformer' ]; then
        batch_size=$(expr 200 \* $mem_scalar)
      elif [ $model = 'transformer' ]; then
        batch_size=$(expr 10 \* $mem_scalar)
      elif [ $model = 'crossformer' ]; then
        batch_size=$(expr 15 \* $mem_scalar)
      fi
      echo torchrun --nproc_per_node=$dev_num --nnodes=1 informermain.py -GBDMC $cuda_device -e $epoch -o $output_len -b $batch_size --fixed_seed 3407 -m $model -d $dataset -S save/23.6.13/$model/$dataset/$output_len
      torchrun --nproc_per_node=$dev_num --nnodes=1 informermain.py -GBDMC $cuda_device -e $epoch -o $output_len -b $batch_size --fixed_seed 3407 -m $model -d $dataset -S save/23.6.13/$model/$dataset/$output_len
    done
  done
done

#for model in informer autoformer fedformer pyraformer transformer reformer; do
#  for dataset in wht gweather etth1 etth2 ettm1 ettm2 exchange ill; do
#    for output_len in 24 48 96 192 360 720; do
#      if [ $model = "reformer" ]; then
#        batch_size=30
#      elif [ $model = 'informer' ]; then
#        batch_size=40
#      elif [ $model = 'autoformer' ]; then
#        batch_size=25
#      elif [ $model = 'fedformer' ]; then
#        batch_size=90
#      elif [ $model = 'pyraformer' ]; then
#        batch_size=200
#      elif [ $model = 'transformer' ]; then
#        batch_size=10
#      elif [ $model = 'crossformer' ]; then
#        batch_size=15
#      fi
#      echo torchrun --nproc_per_node=6 --nnodes=1 informermain.py -GDMC 2,3,4,5,6,7 -e $epoch -o $output_len -b $batch_size --fixed_seed 3407 -m $model -d $dataset -S save/23.6.13/worst_model/$model/$dataset/$output_len
#      torchrun --nproc_per_node=6 --nnodes=1 informermain.py -GDMC 2,3,4,5,6,7 -e $epoch -o $output_len -b $batch_size --fixed_seed 3407 -m $model -d $dataset -S save/23.6.13/worst_model/$model/$dataset/$output_len
#    done
#  done
#done
