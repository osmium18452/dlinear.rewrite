epoch=100
cuda_device=1
exec_date=24.2.5

# 20 40 80 160 320 480
# 24 48 96 192 360 720

len_list="24 48 96 192 360 720"
model=flinear

for dataset in gweather wht etth1 etth2 ettm1 ettm2 exchange; do
    for output_len in $len_list; do
        exec="python main.py -GDBE -C $cuda_device -d $dataset -e $epoch -o $output_len --fixed_seed 3407 -m $model -S save/$exec_date/$model/$dataset/$output_len --fudan"
        echo "$exec"
        $exec
    done
done

for dataset in gweather wht etth1 etth2 ettm1 ettm2 exchange; do
    for output_len in $len_list; do
        exec="python main.py -GDE -C $cuda_device -d $dataset -e $epoch -o $output_len --fixed_seed 3407 -m $model -S save/${exec_date}.not.best.model/$model/$dataset/$output_len --fudan"
        echo "$exec"
        $exec
    done
done


for dataset in ill finance; do
    for output_len in 24 36 48 60; do
        exec="python main.py -GDBE -i 48 -C $cuda_device -d $dataset -e $epoch -o $output_len --fixed_seed 3407 -m $model -S save/$exec_date/$model/$dataset/$output_len --fudan"
        echo "$exec"
        $exec
        exec="python main.py -GDE -i 48 -C $cuda_device -d $dataset -e $epoch -o $output_len --fixed_seed 3407 -m $model -S save/${exec_date}.not.best.model/$model/$dataset/$output_len --fudan"
        echo "$exec"
        $exec
    done
done