##################################################################################################
# 0: Dialog Generation; 1: Knowledge Enhancement; 
# 2: Natural Language Inference; 3: Reinforcement Learning.
##################################################################################################

data=cmu_dog

##################################################################################################
# Stage 0: Baseline GPT-2 dialogue system training and evaluation.
##################################################################################################

if [ $1 == dial ]; then
  echo "$0: Baseline GPT-2 dialogue system training."
    data_path=./../../../data/${data}/dialog_only
    load_path=./models/gpt2-l
    # load_path=./exp/wow/gpt2-l/batch16_epoch4_seq256_lr6e-05
    python -m torch.distributed.launch --nproc_per_node=2 --master_port=9876 codes/dial/main.py \
                                    --data $data \
                                    --stage train \
                                    --batch_size 3 \
                                    --num_epoch 4 \
                                    --gpu_num 2 \
                                    --seq_len 512 \
                                    --learning_rate 6e-5 \
                                    --data_path $data_path \
                                    --cache_path $data_path \
                                    --load_path $load_path \
                                    --know_type none \
                                    --know_size 1024


  echo "$0: Baseline GPT-2 dialogue system evaluation."
    data_path=./../../../data/${data}/dialog_only
    load_path=./exp/${data}/gpt2-l/batch3_epoch4_seq512_lr6e-05_2gpu
    python -m torch.distributed.launch --nproc_per_node=1 --master_port=9876 codes/dial/main.py \
                                    --stage infer \
                                    --data $data \
                                    --data_path $data_path \
                                    --cache_path $data_path \
                                    --load_path $load_path \
                                    --know_type none \
                                    --know_size 1024

fi


##################################################################################################
# Stage 1: Natural language inference pretraining and fine-tuning. Pretraining using gold 
#          wikipedia and response dataset. Fine-tuning on three metrics.
##################################################################################################

if [ $1 == nli ]; then
  echo "$0: Natural language inference pretraining and fine-tuning."
    data_path=./../../../data/wizard_of_wikipedia/conv_fever
    load_path=./exp/nli_bert
    python -m torch.distributed.launch --nproc_per_node=1 --master_port=9876 codes/nli/main.py \
                                    --stage train \
                                    --task fact \
                                    --batch_size 16 \
                                    --num_epoch 5 \
                                    --learning_rate 1e-5 \
                                    --data_path $data_path \
                                    --cache_path $data_path \
                                    --load_path $load_path
fi


##################################################################################################
# Stage 2: Knowledge enhancement for dialogue models. Four strategies are used in this project.
#          1. neural knowledge bank (nkb); 2. K-adapter (kadap); 3. K-former (kform); 
#          4. K-Dialog (kdial).
##################################################################################################

if [ $1 == know ]; then
  echo "$0: Knowledge enhancement for dialogue models."
    data_path=./../../../data/wizard_of_wikipedia/klg_only
    load_path=./models/gpt2-m
    python -m torch.distributed.launch --nproc_per_node=1 --master_port=9876 codes/know/main.py \
                                    --stage train \
                                    --batch_size 16 \
                                    --num_epoch 3 \
                                    --learning_rate 6e-5 \
                                    --data_path $data_path \
                                    --cache_path $data_path \
                                    --load_path $load_path \
                                    --know_type kadp \
                                    --know_size 1024
fi
