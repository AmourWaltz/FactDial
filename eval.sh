##################################################################################################
# 0: Dialog Generation; 1: Knowledge Enhancement; 2: Natural Language Inference; 3: Reinforcement Learning.
##################################################################################################

data=cmu_dog

##################################################################################################
# Stage 0: Baseline GPT-2 dialogue system training and evaluation.
##################################################################################################

if [ $1 == dial ]; then
  echo "$0: Baseline GPT-2 dialogue system training and evaluation."
    data_path=./../../../data/${data}/dialog_only
    load_path=./exp/${data}/gpt2-m/batch8_epoch3_seq448_lr6e-05
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
    # data_path=./../../../data/wizard_of_wikipedia/conv_fever
    nli_task=verif
    data_path=/home/ma-user/work/byxue/project/dialog/fact_dial/exp/gpt2-l_batch6_epoch3_seq256_lr6e-05/beam1_sampFalse_temp0.0_tk0.0_tp0.0
    load_path=./exp/nli_bert_batch16_epoch5_seq512_lr1e-05_ratio0.5_${nli_task}
    # load_path=./exp/nli_bert
    python -m torch.distributed.launch --nproc_per_node=1 --master_port=9876 codes/nli/main.py \
                                    --stage score \
                                    --task $nli_task \
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
    load_path=./exp/gpt2-l_batch6_epoch3_seq256_lr6e-05
    python -m torch.distributed.launch --nproc_per_node=1 --master_port=9876 codes/know/main.py \
                                    --stage eval \
                                    --model_type gpt2-m \
                                    --data_path $data_path \
                                    --cache_path $data_path \
                                    --load_path $load_path \
                                    --know_type nkb
fi
