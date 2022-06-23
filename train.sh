#!/bin/bash
#
# Automatically change the parameter
#
# 

num_classes="55:54:52:56"
batch_sizes="256:128"
gaussian_sizes="48:32"

cd $(dirname $0)


dataset=$1
[ -d result_save/$dataset ] || mkdir -p result_save/$dataset



# available_devices=("0" "1" "2" "3" "4" "5" "6" "7")

main() {

    count=0
    for num_class in ${num_classes//:/ }; do
    for batch_size in ${batch_sizes//:/ }; do
    for gaussian_size in ${gaussian_sizes//:/ }; do
    {
        cuda_id=$(($count % 1))
        # available_devices_inloop=${available_devices[$cuda_id]}

        call > result_save/$dataset/bins$num_class-batch_size$batch_size-gaussian_size$gaussian_size.log 2>&1 &
        if [[ $((($count+1) % 1)) -eq 0 ]]; then
            wait
        fi
    }
    let count+=1
    done
    done
    done
}

call() {
    # --shrinkage_loss \
    # --shrinkage_loss_a $shrinkage_loss_a \
    # mkdir -p $log_path/$exp_name-$count
    /home/comp/zmzhang/software/bin/time -v python main.py \
    --num_classes $num_class \
    --batch_size $batch_size \
    --gaussian_size $gaussian_size

}


main
