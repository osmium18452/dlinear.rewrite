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
# o: output length 24 48 96 192 360 720
#    or 20 40 80 160 320 480
# seven transformer-like models, seven datasets and six output lengths.
# illness dataset: 24 36 48 60
epoch=20
mem_scalar=1
cuda_device=3

model_list="crossformer informer autoformer fedformer pyraformer transformer reformer"
out_list="24 36 48 60"
exec_date=23.8.9
prefix=finance.sup
flags="-GD"

for model in $model_list; do
    dataset='finance'
    for output_len in $out_list; do
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
            batch_size=$(expr 10 \* $mem_scalar)
        fi
        exec="python informermain.py $flags --fudan -C $cuda_device -e $epoch -o $output_len -b $batch_size --fixed_seed 3407 -m $model -d $dataset -S save/$prefix/$exec_date/$model/$dataset/$output_len"
        echo "$exec"
        if [ ! -e save/$prefix/$exec_date/$model/$dataset/$output_len/result.json ]; then
            $exec
        else
            echo 'file exists'
        fi
    done
done
