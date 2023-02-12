#!/bin/bash
# dim_latent, l_r, reg_parm을 다르게 주는 코드
# number를 통해 각 파일을 구분해줌


# for model_name in GRAPH_v GRAPH_t GRAPH_p GRAPH_tr
# do 
#     python -u train.py \
#         --num_epoch 50 \
#         --l_r 0.01 \
#         --dim_latent 128 \
#         --reg_parm 1e-5 \
#         --batch_size 2048 \
#         --weight_decay 0.1 \
#         --data_path dataset/azuki/ \
#         --neg_sample 5 \
#         --PATH_weight_save model/gg/ \
#         --loss_alpha 0.1 \
#         --model_name $model_name &
# done

# sleep 3600

# for model_name in GRAPH_v GRAPH_t GRAPH_p GRAPH_tr
# do 
#     python -u train.py \
#         --num_epoch 50 \
#         --l_r 0.001 \
#         --dim_latent 512 \
#         --reg_parm 1e-5 \
#         --batch_size 2048 \
#         --weight_decay 0.1 \
#         --data_path dataset/bayc/ \
#         --neg_sample 5 \
#         --PATH_weight_save model/gg/ \
#         --loss_alpha 0.1 \
#         --model_name $model_name &
# done

# sleep 3600


# for model_name in GRAPH_v GRAPH_t GRAPH_p GRAPH_tr
# do 
#     python -u train.py \
#         --num_epoch 50 \
#         --l_r 0.01 \
#         --dim_latent 256 \
#         --reg_parm 0 \
#         --batch_size 2048 \
#         --weight_decay 0.1 \
#         --data_path dataset/coolcats/ \
#         --neg_sample 5 \
#         --PATH_weight_save model/gg/ \
#         --loss_alpha 0.1 \
#         --model_name $model_name &

# done

# sleep 3600


# for model_name in GRAPH_v GRAPH_t GRAPH_p GRAPH_tr
# do 
#     python -u train.py \
#         --num_epoch 50 \
#         --l_r 0.01 \
#         --dim_latent 512 \
#         --reg_parm 1e-5 \
#         --batch_size 2048 \
#         --weight_decay 0.1 \
#         --data_path dataset/doodles/ \
#         --neg_sample 5 \
#         --PATH_weight_save model/gg/ \
#         --loss_alpha 0.1 \
#         --model_name $model_name &

# done

# sleep 3600


for model_name in GRAPH_v GRAPH_t GRAPH_p GRAPH_tr
do 
    python -u train.py \
        --num_epoch 50 \
        --l_r 0.01 \
        --dim_latent 128 \
        --reg_parm 0 \
        --batch_size 2048 \
        --weight_decay 0.1 \
        --data_path dataset/meebits/ \
        --neg_sample 5 \
        --PATH_weight_save model/gg/ \
        --loss_alpha 0.1 \
        --model_name $model_name &

done