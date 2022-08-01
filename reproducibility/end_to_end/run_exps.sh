export NAUTILUS_RANDOM_SEED=2020
export CUDA_VISBLE_DEVICES=0
device='gpu'

for iter in 1 2 3
do
    mode='ftu'
    python ../../examples/malaria_ftu.py > ${mode}/${iter}-${device}-mat-fuse-raw.log 2> /dev/null
    cat ${mode}/${iter}-${device}-mat-fuse-raw.log | grep NAUTILUS > ${mode}/${iter}-${device}-mat-fuse.log

    python ../../examples/malaria_ftu.py --no-mat-opt > ${mode}/${iter}-${device}-no-mat-fuse-raw.log 2> /dev/null
    cat ${mode}/${iter}-${device}-no-mat-fuse-raw.log | grep NAUTILUS > ${mode}/${iter}-${device}-no-mat-fuse.log

    python ../../examples/malaria_ftu.py --no-fuse-opt > ${mode}/${iter}-${device}-mat-no-fuse-raw.log 2> /dev/null
    cat ${mode}/${iter}-${device}-mat-no-fuse-raw.log | grep NAUTILUS > ${mode}/${iter}-${device}-mat-no-fuse.log

    python ../../examples/malaria_ftu.py --disk-throughput 999999999999 --no-fuse-opt > ${mode}/${iter}-${device}-mat-all-no-fuse-raw.log 2> /dev/null
    cat ${mode}/${iter}-${device}-mat-all-no-fuse-raw.log | grep NAUTILUS > ${mode}/${iter}-${device}-mat-all-no-fuse.log

    python ../../examples/malaria_ftu.py --no-mat-opt --no-fuse-opt > ${mode}/${iter}-${device}-no-mat-no-fuse-raw.log 2> /dev/null
    cat ${mode}/${iter}-${device}-no-mat-no-fuse-raw.log | grep NAUTILUS > ${mode}/${iter}-${device}-no-mat-no-fuse.log


    mode='atr'
    python ../../examples/conll_atr.py > ${mode}/${iter}-${device}-mat-fuse-raw.log 2> /dev/null
    cat ${mode}/${iter}-${device}-mat-fuse-raw.log | grep NAUTILUS > ${mode}/${iter}-${device}-mat-fuse.log

    python ../../examples/conll_atr.py --no-mat-opt > ${mode}/${iter}-${device}-no-mat-fuse-raw.log 2> /dev/null
    cat ${mode}/${iter}-${device}-no-mat-fuse-raw.log | grep NAUTILUS > ${mode}/${iter}-${device}-no-mat-fuse.log

    python ../../examples/conll_atr.py --no-fuse-opt > ${mode}/${iter}-${device}-mat-no-fuse-raw.log 2> /dev/null
    cat ${mode}/${iter}-${device}-mat-no-fuse-raw.log | grep NAUTILUS > ${mode}/${iter}-${device}-mat-no-fuse.log

    python ../../examples/conll_atr.py --disk-throughput 999999999999 --no-fuse-opt > ${mode}/${iter}-${device}-mat-all-no-fuse-raw.log 2> /dev/null
    cat ${mode}/${iter}-${device}-mat-all-no-fuse-raw.log | grep NAUTILUS > ${mode}/${iter}-${device}-mat-all-no-fuse.log

    python ../../examples/conll_atr.py --no-mat-opt --no-fuse-opt > ${mode}/${iter}-${device}-no-mat-no-fuse-raw.log 2> /dev/null
    cat ${mode}/${iter}-${device}-no-mat-no-fuse-raw.log | grep NAUTILUS > ${mode}/${iter}-${device}-no-mat-no-fuse.log


    for mode in 'ftr1' 'ftr2' 'ftr3'
    do
       python ../../examples/conll_ftr.py --mode ${mode} > ${mode}/${iter}-${device}-mat-fuse-raw.log 2> /dev/null
       cat ${mode}/${iter}-${device}-mat-fuse-raw.log | grep NAUTILUS > ${mode}/${iter}-${device}-mat-fuse.log

       python ../../examples/conll_ftr.py --mode ${mode} --no-mat-opt > ${mode}/${iter}-${device}-no-mat-fuse-raw.log 2> /dev/null
       cat ${mode}/${iter}-${device}-no-mat-fuse-raw.log | grep NAUTILUS > ${mode}/${iter}-${device}-no-mat-fuse.log

       python ../../examples/conll_ftr.py --mode ${mode} --no-fuse-opt > ${mode}/${iter}-${device}-mat-no-fuse-raw.log 2> /dev/null
       cat ${mode}/${iter}-${device}-mat-no-fuse-raw.log | grep NAUTILUS > ${mode}/${iter}-${device}-mat-no-fuse.log

       python ../../examples/conll_ftr.py --mode ${mode} --disk-throughput 999999999999 --no-fuse-opt > ${mode}/${iter}-${device}-mat-all-no-fuse-raw.log 2> /dev/null
       cat ${mode}/${iter}-${device}-mat-all-no-fuse-raw.log | grep NAUTILUS > ${mode}/${iter}-${device}-mat-all-no-fuse.log
       
       python ../../examples/conll_ftr.py --mode ${mode} --no-mat-opt --no-fuse-opt > ${mode}/${iter}-${device}-no-mat-no-fuse-raw.log 2> /dev/null
       cat ${mode}/${iter}-${device}-no-mat-no-fuse-raw.log | grep NAUTILUS > ${mode}/${iter}-${device}-no-mat-no-fuse.log
    done

done

rm -r storage
