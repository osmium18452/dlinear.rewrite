epoch=20
mem_scalar=1
cuda_device=0,1,2,3,4,5,6,7
dev_num=8

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
        exec="torchrun --nproc_per_node=$dev_num --nnodes=1 informermain.py -GDMC $cuda_device -e $epoch -o $output_len -b $batch_size --fixed_seed 3407 -m $model -d $dataset -S save/$exec_date/$model/$dataset/$output_len"
        echo "$exec"
        $exec
    done
done