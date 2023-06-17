lr_list="0.01 0.001 0.0001 0.00005"
output_list="720 20 24 40 48 80 96 160 192 320 360 480"
epochs=50
exec_date=23.6.17
dev_num=8
batch_size=32

for lr in $lr_list; do
    for output_len in $output_list; do
        exec="torchrun --nproc_per_node=$dev_num --nnodes=1 cgatmain.py -GBDM -e $epochs -o $output_len -b $batch_size --fixed_seed 3407 -d wht -S save/$exec_date/not.ind/$lr/$output_len -l $lr"
        echo $exec
        if [ ! -e save/$exec_date/not.ind/$lr/$output_len/result.json ]; then
            $exec
        else
            echo 'file exists'
        fi
        exec="torchrun --nproc_per_node=$dev_num --nnodes=1 cgatmain.py -GBDM -e $epochs -o $output_len -b $batch_size --fixed_seed 3407 -d wht -S save/$exec_date/individual/$lr/$output_len -l $lr -I"
        echo $exec
        if [ ! -e save/$exec_date/individual/$lr/$output_len/result.json ]; then
            $exec
        else
            echo 'file exists'
        fi
    done
done
