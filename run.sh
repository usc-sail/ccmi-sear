#!/bin/bash


#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB
#SBATCH --time=05:00:00
#SBATCH --partition=gpu 
#SBATCH --gres=gpu:p100:1
#SBATCH --account=shrikann_35


#module load gcc/8.3.0
#module load libsndfile/1.0.31
#module load cuda/11.4.0
#module load cudnn/8.2.4.15-11.4
#module load nvidia-hpc-sdk
#module load libsndfile
eval "$(conda shell.bash hook)"
conda activate /home1/rajatheb/.conda/envs/ssasta100


#pretrain_path=/scratch1/rajatheb/ssl_icassp/pretrained_models/ssast-base-m250-b90.pth
pretrain_path=/scratch1/rajatheb/ssl_icassp/pretrained_models/SSAST-Base-Patch-250.pth
#pretrain_path=/scratch1/rajatheb/ssl_icassp/pretrained_models/combined_old_splits/combined_newsplit/ast-combn-clip-b170-lr1e-4-w12k.pth
#pretrain_path=/scratch1/rajatheb/ssl_icassp/pretrained_models/combined/ast-comb-clip-b180-lr1e-4-w12k.pth
#pretrain_path=None
#pretrain_path=/scratch1/rajatheb/ssl_icassp/pretrained_models/combined/ast-comb-clipmasked-b150-lr1e-4-w12k.pth
#pretrain_path=/scratch1/rajatheb/ssl_icassp/pretrained_models/audioset/ast-as-clip-b180-lr1e-4.pth

#pretrain_path=None

pretrain_model=$(basename $pretrain_path .pth)
dataset=esc50
dataset_mean=-6.6268077
dataset_std=5.358466
target_length=512
noise=True

bal=none
lr=1e-4
freqm=48
timem=96
mixup=0
epoch=30
batch_size=48
fshape=16
tshape=16
fstride=16
tstride=16

task=ft_cls
model_size=base
head_lr=3

input_prompt=False
embedding_prompt=True
adapter=False
num_inp_prompts=10
num_emb_prompts=2

base_exp_dir=./exp_${task}/ip${input_prompt}-${num_inp_prompts}_ep${embedding_prompt}-${num_emb_prompts}_ad${adapter}/${pretrain_model}_hlr${head_lr}_nooverlap
#base_exp_dir=./exp_${task}/${pretrain_model}_hlr${head_lr}

if [ -d $base_exp_dir ]; then
    echo 'exp exist'
    exit
fi
mkdir -p ${base_exp_dir}

for((fold=1;fold<=5;fold++));
do
  echo 'now process fold'${fold}

  exp_dir=${base_exp_dir}/fold${fold}
  mkdir -p $exp_dir/models

  tr_data=./data/esc_train_data_${fold}.json
  te_data=./data/esc_eval_data_${fold}.json

  CUDA_CACHE_DISABLE=1 python -W ignore ../../run_ft.py --dataset ${dataset} \
  --data-train ${tr_data} --data-val ${te_data} --exp-dir $exp_dir \
  --label-csv ./data/esc_class_labels_indices.csv --n_class 50 \
  --lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model False \
  --freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
  --tstride $tstride --fstride $fstride --fshape ${fshape} --tshape ${tshape} --warmup True --task ${task} \
  --model_size ${model_size} --adaptschedule False \
  --pretrained_mdl_path ${pretrain_path} \
  --dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --target_length ${target_length} \
  --num_mel_bins 128 --head_lr ${head_lr} --noise ${noise} \
  --input_prompt ${input_prompt} --embedding_prompt ${embedding_prompt} --adapter ${adapter} \
  --num_inp_prompts ${num_inp_prompts} --num_emb_prompts ${num_emb_prompts} \
  --lrscheduler_start 6 --lrscheduler_step 1 --lrscheduler_decay 0.95 --wa False --loss CE --metrics acc --n-print-steps 10 > ${exp_dir}/log.txt
done

python ./get_esc_result.py --exp_path ${base_exp_dir}

