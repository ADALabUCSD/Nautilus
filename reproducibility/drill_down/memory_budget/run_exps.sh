mkdir -p logs
export NAUTILUS_RANDOM_SEED=2020
export CUDA_VISBLE_DEVICES=0
device='gpu'
mode='ftr2'

for iter in 1 2 3
do
    for mem_budget in 2 4 6 8 10
    do
        python ../../../examples/conll_ftr.py --mode ${mode} --no-mat-opt --memory-budget $mem_budget > logs/${iter}-${device}-${mem_budget}-no-mat-fuse-raw.log 2> /dev/null
        cat logs/${iter}-${device}-${mem_budget}-no-mat-fuse-raw.log | grep NAUTILUS > logs/${iter}-${device}-${mem_budget}-no-mat-fuse.log
    done
done

rm -r storage

