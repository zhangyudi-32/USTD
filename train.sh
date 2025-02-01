model=$1
dataset=$2
attr=$3
t_len=$4
pretrain=$5
config=$6
batch=$7
gpu_ids=$8

echo gpu_ids $gpu_ids

python train.py --model ${model}\
  --dataset_mode ${dataset}\
  --pred_attr ${attr}\
  --enable_val\
  --gpu_ids ${gpu_ids}\
  --config ${config}\
  --pretrain ${pretrain}\
  --save_best\
  --t_len ${t_len}\
  --eval_epoch_freq 1\
  --num_train_target 3\
  --num_threads 4\
  --batch_size ${batch}
# done
