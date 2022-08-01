export NAUTILUS_RANDOM_SEED=2020
export CUDA_VISBLE_DEVICES=0
device='gpu'
mode='ftr2'

mkdir -p logs

for iter in 1 2 3
do
    # ############################################## FTR ###############################################
    for storage_budget in 10 7.5 5 2.5 0
    do
        python ../../../examples/conll_ftr.py --mode ${mode} --no-fuse-opt --storage-budget $storage_budget > logs/${iter}-${device}-${storage_budget}-mat-no-fuse-raw.log 2> /dev/null
        cat logs/${iter}-${device}-${storage_budget}-mat-no-fuse-raw.log | grep NAUTILUS > logs/${iter}-${device}-${storage_budget}-mat-no-fuse.log

    done
done

rm -r storage
