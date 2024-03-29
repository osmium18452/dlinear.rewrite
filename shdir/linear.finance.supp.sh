epoch=20
cuda_device=3
exec_date=24.2.4


for model in linear dlinear nlinear; do
    for output_len in 24 36 48 60; do
        dataset=finance
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