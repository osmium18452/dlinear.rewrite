torchrun --nproc_per_node=2 --nnodes=1 informermain.py -GDM -e 20 -o 320 -b 10 --fixed_seed 3407 -m fedformer -d ettm2 -S save/formers/fedformer/ettm2/320
torchrun --nproc_per_node=2 --nnodes=1 informermain.py -GDM -e 20 -o 480 -b 10 --fixed_seed 3407 -m fedformer -d ettm2 -S save/formers/fedformer/ettm2/480

torchrun --nproc_per_node=2 --nnodes=1 informermain.py -GDM -e 20 -o 20  -b 10 --fixed_seed 3407 -m fedformer -d exchange -S save/formers/fedformer/exchange/20
torchrun --nproc_per_node=2 --nnodes=1 informermain.py -GDM -e 20 -o 40  -b 10 --fixed_seed 3407 -m fedformer -d exchange -S save/formers/fedformer/exchange/40
torchrun --nproc_per_node=2 --nnodes=1 informermain.py -GDM -e 20 -o 80  -b 10 --fixed_seed 3407 -m fedformer -d exchange -S save/formers/fedformer/exchange/80
torchrun --nproc_per_node=2 --nnodes=1 informermain.py -GDM -e 20 -o 160 -b 10 --fixed_seed 3407 -m fedformer -d exchange -S save/formers/fedformer/exchange/160
torchrun --nproc_per_node=2 --nnodes=1 informermain.py -GDM -e 20 -o 320 -b 10 --fixed_seed 3407 -m fedformer -d exchange -S save/formers/fedformer/exchange/320
torchrun --nproc_per_node=2 --nnodes=1 informermain.py -GDM -e 20 -o 480 -b 10 --fixed_seed 3407 -m fedformer -d exchange -S save/formers/fedformer/exchange/480

torchrun --nproc_per_node=2 --nnodes=1 informermain.py -GDM -e 20 -o 20  -b 10 --fixed_seed 3407 -m fedformer -d wht -S save/formers/fedformer/wht/20
torchrun --nproc_per_node=2 --nnodes=1 informermain.py -GDM -e 20 -o 40  -b 10 --fixed_seed 3407 -m fedformer -d wht -S save/formers/fedformer/wht/40
torchrun --nproc_per_node=2 --nnodes=1 informermain.py -GDM -e 20 -o 80  -b 10 --fixed_seed 3407 -m fedformer -d wht -S save/formers/fedformer/wht/80
torchrun --nproc_per_node=2 --nnodes=1 informermain.py -GDM -e 20 -o 160 -b 10 --fixed_seed 3407 -m fedformer -d wht -S save/formers/fedformer/wht/160
torchrun --nproc_per_node=2 --nnodes=1 informermain.py -GDM -e 20 -o 320 -b 10 --fixed_seed 3407 -m fedformer -d wht -S save/formers/fedformer/wht/320
torchrun --nproc_per_node=2 --nnodes=1 informermain.py -GDM -e 20 -o 480 -b 10 --fixed_seed 3407 -m fedformer -d wht -S save/formers/fedformer/wht/480
