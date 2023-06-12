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
epoch=3

for dataset in wht gweather etth1 etth2 ettm1 ettm2 exchange ill; do
  for model in informer autoformer fedformer pyraformer transformer reformer linear dlinear nlinear; do
    for output_len in 24 48 96 192 360 720; do
      if [ $model = "reformer" ]; then
        batch_size=30
      elif [ $model = 'informer' ]; then
        batch_size=40
      elif [ $model = 'autoformer' ]; then
        batch_size=25
      elif [ $model = 'fedformer' ]; then
        batch_size=90
      elif [ $model = 'pyraformer' ]; then
        batch_size=200
      elif [ $model = 'transformer' ]; then
        batch_size=10
      elif [ $model = 'crossformer' ]; then
        batch_size=15
      fi
      echo torchrun --nproc_per_node=6 --nnodes=1 informermain.py -GBDMC 2,3,4,5,6,7 -e $epoch -o $output_len -b $batch_size --fixed_seed 3407 -m $model -d $dataset -S save/23.6.13/$model/$dataset/$output_len
      torchrun --nproc_per_node=6 --nnodes=1 informermain.py -GBDMC 2,3,4,5,6,7 -e $epoch -o $output_len -b $batch_size --fixed_seed 3407 -m $model -d $dataset -S save/23.6.13/$model/$dataset/$output_len
    done
  done
done

for dataset in wht gweather etth1 etth2 ettm1 ettm2 exchange ill; do
  for model in informer autoformer fedformer pyraformer transformer reformer linear dlinear nlinear; do
    for output_len in 24 48 96 192 360 720; do
      if [ $model = "reformer" ]; then
        batch_size=30
      elif [ $model = 'informer' ]; then
        batch_size=40
      elif [ $model = 'autoformer' ]; then
        batch_size=25
      elif [ $model = 'fedformer' ]; then
        batch_size=90
      elif [ $model = 'pyraformer' ]; then
        batch_size=200
      elif [ $model = 'transformer' ]; then
        batch_size=10
      elif [ $model = 'crossformer' ]; then
        batch_size=15
      fi
      echo torchrun --nproc_per_node=6 --nnodes=1 informermain.py -GDMC 2,3,4,5,6,7 -e $epoch -o $output_len -b $batch_size --fixed_seed 3407 -m $model -d $dataset -S save/23.6.13/worst_model/$model/$dataset/$output_len
      torchrun --nproc_per_node=6 --nnodes=1 informermain.py -GDMC 2,3,4,5,6,7 -e $epoch -o $output_len -b $batch_size --fixed_seed 3407 -m $model -d $dataset -S save/23.6.13/worst_model/$model/$dataset/$output_len
    done
  done
done