mkdir -p logs
export NAUTILUS_RANDOM_SEED=2020
export CUDA_VISBLE_DEVICES=0
device='gpu'

for iter in 1 2 3
do

    # ############################################## FTR ###############################################
    for mode in 'ftr3-1' 'ftr3-2' 'ftr3-3' 'ftr3-4' 'ftr3-5'
    do

        python ../../../examples/conll_ftr.py --mode ${mode} > logs/${iter}-${device}-${mode}-mat-fuse-raw.log 2> /dev/null
        cat logs/${iter}-${device}-${mode}-mat-fuse-raw.log | grep NAUTILUS > logs/${iter}-${device}-${mode}-mat-fuse.log

        python ../../../examples/conll_ftr.py --mode ${mode} --no-mat-opt > logs/${iter}-${device}-${mode}-no-mat-fuse-raw.log 2> /dev/null
        cat logs/${iter}-${device}-${mode}-no-mat-fuse-raw.log | grep NAUTILUS > logs/${iter}-${device}-${mode}-no-mat-fuse.log

        python ../../../examples/conll_ftr.py --mode ${mode} --no-fuse-opt > logs/${iter}-${device}-${mode}-mat-no-fuse-raw.log 2> /dev/null
        cat logs/${iter}-${device}-${mode}-mat-no-fuse-raw.log | grep NAUTILUS > logs/${iter}-${device}-${mode}-mat-no-fuse.log

    done

done

rm -r storage
