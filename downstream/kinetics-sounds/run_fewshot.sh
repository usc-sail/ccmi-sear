#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB
#SBATCH --time=12:00:00
#SBATCH --partition=gpu 
#SBATCH --gres=gpu:a100:1
#SBATCH --account=shrikann_35
# comm SBATCH --exclude=b04-[09-11],b05-[09-11],b09-[09-14],b11-[09-14],b10-[09-11]

#module load gcc/11.3.0
#module load libsndfile
#module load cuda/11.4.0
#module load cudnn/8.2.4.15-11.4
#module load nvidia-hpc-sdk
eval "$(conda shell.bash hook)"
conda activate /home1/rajatheb/.conda/envs/ssasta100


#pretrain_path=/scratch1/rajatheb/ssl_icassp/pretrained_models/clip/ast-base-clip-b90-w12k-decay0.9.pth
#pretrain_path=/scratch1/rajatheb/ssl_icassp/pretrained_models/combined/ast-comb-clip-b180-lr1e-4-w12k.pth
#pretrain_path=/scratch1/rajatheb/ssl_icassp/pretrained_models/audioset/ast-as-clip-b180-lr1e-4.pth
pretrain_path=/scratch1/rajatheb/ssl_icassp/pretrained_models/SSAST-Base-Patch-250.pth
#pretrain_path=None

pretrain_model=$(basename $pretrain_path .pth)

dataset=kinetics_sounds
dataset_mean=-5.444231
dataset_std=3.2999249
target_length=1024
noise=False
tr_data=./data/train.json
val_data=./data/val.json
test_data=./data/test.json

fewshot_mode=raw
fewshot_numsamples=20

bal=none
lr=1e-4
freqm=48
timem=192
mixup=0
epoch=25
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
num_emb_prompts=16

#pretrain_exp=./exp_${task}/ip${input_prompt}-${num_inp_prompts}_ep${embedding_prompt}-${num_emb_prompts}_ad${adapter}/${pretrain_model}_hlr${head_lr}_nooverlap
pretrain_exp=./exp_fewshot/${pretrain_model}_lr${lr}_hlr${head_lr}/mode-${fewshot_mode}_${fewshot_numsamples}
#pretrain_exp=./exp-cm_${task}/${pretrain_model}_hlr${head_lr}

mkdir -p $pretrain_exp
CUDA_CACHE_DISABLE=1 python -W ignore ../../run_ft.py --dataset ${dataset} \
--data-train ${tr_data} --data-val ${val_data} --data-eval ${test_data} --exp-dir $pretrain_exp \
--label-csv ./data/kinetics_sounds_classes.csv --n_class 29 \
--lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model True \
--fewshot_mode ${fewshot_mode} --fewshot_num_samples ${fewshot_numsamples} \
--freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
--tstride $tstride --fstride $fstride --fshape ${fshape} --tshape ${tshape} --warmup True --task ${task} \
--model_size ${model_size} --adaptschedule False \
--pretrained_mdl_path ${pretrain_path} \
--dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --target_length ${target_length} \
--input_prompt ${input_prompt} --embedding_prompt ${embedding_prompt} --adapter ${adapter} \
--num_inp_prompts ${num_inp_prompts} --num_emb_prompts ${num_emb_prompts} \
--num_mel_bins 128 --head_lr ${head_lr} --noise ${noise} --wa_start 10 --wa_end 25 \
--lrscheduler_start 10 --lrscheduler_step 1 --lrscheduler_decay 0.85 --wa True --loss CE --metrics acc > $pretrain_exp/log.txt

