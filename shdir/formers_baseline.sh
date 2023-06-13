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
#    or 20 40 80 160 320
# seven transformer-like models, seven datasets and six output lengths.
# illness dataset: 24 36 48 60
epoch=20
mem_scalar=2
cuda_device=1,2,4,5,6,7
dev_num=6

out_list="24 48 96 192 360 720"
exec_date=23.6.13.1.worst.model

# wht complement

for model in crossformer informer autoformer fedformer pyraformer transformer reformer; do
    dataset=wht
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
        if [ -e save/formers/$exec_date/$model/$dataset/$output_len/result.json ]; then
            continue
        fi
        exec="torchrun --nproc_per_node=$dev_num --nnodes=1 informermain.py --fudan -GDMC $cuda_device -e $epoch -o $output_len -b $batch_size --fixed_seed 3407 -m $model -d $dataset -S save/formers/$exec_date/$model/$dataset/$output_len"
        echo "$exec"
        $exec
    done
done

exec_date=23.6.13.1

for model in crossformer informer autoformer fedformer pyraformer transformer reformer; do
    dataset=wht
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
        if [ -e save/formers/$exec_date/$model/$dataset/$output_len/result.json ]; then
            continue
        fi
        exec="torchrun --nproc_per_node=$dev_num --nnodes=1 informermain.py --fudan -GBDMC $cuda_device -e $epoch -o $output_len -b $batch_size --fixed_seed 3407 -m $model -d $dataset -S save/formers/$exec_date/$model/$dataset/$output_len"
        echo "$exec"
        $exec
    done
done

# run other lengths

out_list="20 40 80 160 320 480"
exec_date=23.6.14.worst.model

for model in crossformer informer autoformer fedformer pyraformer transformer reformer; do
    for dataset in gweather etth1 etth2 ettm1 ettm2 exchange wht; do
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
            if [ -e save/formers/$exec_date/$model/$dataset/$output_len/result.json ]; then
                continue
            fi
            exec="torchrun --nproc_per_node=$dev_num --nnodes=1 informermain.py --fudan -GDMC $cuda_device -e $epoch -o $output_len -b $batch_size --fixed_seed 3407 -m $model -d $dataset -S save/formers/$exec_date/$model/$dataset/$output_len"
            echo "$exec"
            $exec
        done
    done
done

exec_date=23.6.14

for model in crossformer informer autoformer fedformer pyraformer transformer reformer; do
    for dataset in gweather etth1 etth2 ettm1 ettm2 exchange wht; do
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
            if [ -e save/formers/$exec_date/$model/$dataset/$output_len/result.json ]; then
                continue
            fi
            exec="torchrun --nproc_per_node=$dev_num --nnodes=1 informermain.py --fudan -GBDMC $cuda_device -e $epoch -o $output_len -b $batch_size --fixed_seed 3407 -m $model -d $dataset -S save/formers/$exec_date/$model/$dataset/$output_len"
            echo "$exec"
            $exec
        done
    done
done
