#!/bin/sh
epoch=20
cuda_device=3
exec_date=23.6.13.2

# 20 40 80 160 320 480
# 24 48 96 192 360 720

len_list="20 40 80 160 320 480 24 48 96 192 360 720"

for model in linear dlinear nlinear; do
    for dataset in gweather wht etth1 etth2 ettm1 ettm2 exchange; do
        for output_len in $len_list; do
            exec="python main.py -GDBE -C $cuda_device -d $dataset -e $epoch -o $output_len --fixed_seed 3407 -m $model -S save/$exec_date/$model/$dataset/$output_len --fudan"
            echo "$exec"
            $exec
        done
    done
done

for model in linear dlinear nlinear; do
    for dataset in gweather wht etth1 etth2 ettm1 ettm2 exchange; do
        for output_len in $len_list; do
            exec="python main.py -GDBEI -C $cuda_device -d $dataset -e $epoch -o $output_len --fixed_seed 3407 -m $model -S save/$exec_date/${model}.individual/$dataset/$output_len --fudan"
            echo "$exec"
            $exec
        done
    done
done

for model in linear dlinear nlinear; do
    for dataset in gweather wht etth1 etth2 ettm1 ettm2 exchange; do
        for output_len in $len_list; do
            exec="python main.py -GDE -C $cuda_device -d $dataset -e $epoch -o $output_len --fixed_seed 3407 -m $model -S save/${exec_date}.not.best.model/$model/$dataset/$output_len --fudan"
            echo "$exec"
            $exec
        done
    done
done

for model in linear dlinear nlinear; do
    for dataset in gweather wht etth1 etth2 ettm1 ettm2 exchange; do
        for output_len in $len_list; do
            exec="python main.py -GDEI -C $cuda_device -d $dataset -e $epoch -o $output_len --fixed_seed 3407 -m $model -S save/${exec_date}.not.best.model/${model}.individual/$dataset/$output_len --fudan"
            echo "$exec"
            $exec
        done
    done
done

for model in linear dlinear nlinear; do
    for output_len in 24 36 48 60; do
        dataset=ill
        exec="python main.py -GDBE -C $cuda_device -d $dataset -e $epoch -o $output_len --fixed_seed 3407 -m $model -S save/$exec_date/$model/$dataset/$output_len --fudan"
        echo "$exec"
        $exec
        exec="python main.py -GDBEI -C $cuda_device -d $dataset -e $epoch -o $output_len --fixed_seed 3407 -m $model -S save/$exec_date/${model}.individual/$dataset/$output_len --fudan"
        echo "$exec"
        $exec
        exec="python main.py -GDE -C $cuda_device -d $dataset -e $epoch -o $output_len --fixed_seed 3407 -m $model -S save/${exec_date}.not.best.model/$model/$dataset/$output_len --fudan"
        echo "$exec"
        $exec
        exec="python main.py -GDEI -C $cuda_device -d $dataset -e $epoch -o $output_len --fixed_seed 3407 -m $model -S save/${exec_date}.not.best.model/${model}.individual/$dataset/$output_len --fudan"
        echo "$exec"
        $exec
    done
done
