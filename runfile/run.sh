# !/bin/bash
# dim_latent, l_r, reg_parm을 다르게 주는 코드
# number를 통해 각 파일을 구분해줌

number=0

for attention_dropout in 0.2 0.4 0.6
do 

    python -u main.py \
        --number $number \
        --num_epoch 50 \
        --dim_latent 128 \
        --batch_size 2048 \
        --weight_decay 0.1 \
        --l_r 0.001 \
        --data_path dataset/collections/coolcats/ \
        --reg_parm 1e-5 \
        --neg_sample 10 \
        --PATH_weight_save model/coolcats_1/ \
        --loss_alpha 0.2 \
        --attention_dropout $attention_dropout \
        --model_name MGAT_2 &

    let number=$number+1


done
wait

for attention_dropout in 0.2 0.4 0.6
do 

    python -u train.py \
        --number $number \
        --num_epoch 50 \
        --dim_latent 256 \
        --batch_size 2048 \
        --weight_decay 0.1 \
        --l_r 0.001 \
        --data_path dataset/coolcats/ \
        --reg_parm 1e-5 \
        --neg_sample 10 \
        --PATH_weight_save model/coolcats_1/ \
        --loss_alpha 0.2 \
        --attention_dropout $attention_dropout \
        --model_name MGAT_2 &

    let number=$number+1


done
wait


for attention_dropout in 0.2 0.4 0.6
do 

    python -u train.py \
        --number $number \
        --num_epoch 50 \
        --dim_latent 512 \
        --batch_size 2048 \
        --weight_decay 0.1 \
        --l_r 0.001 \
        --data_path dataset/coolcats/ \
        --reg_parm 1e-5 \
        --neg_sample 10 \
        --PATH_weight_save model/coolcats_1/ \
        --loss_alpha 0.2 \
        --attention_dropout $attention_dropout \
        --model_name MGAT_2 &

    let number=$number+1


done
wait
number=0

for attention_dropout in 0.2 0.4 0.6
do 

    python -u train.py \
        --number $number \
        --num_epoch 50 \
        --dim_latent 128 \
        --batch_size 2048 \
        --weight_decay 0.1 \
        --l_r 0.001 \
        --data_path dataset/coolcats2/ \
        --reg_parm 1e-5 \
        --neg_sample 10 \
        --PATH_weight_save model/coolcats_2/ \
        --loss_alpha 0.2 \
        --attention_dropout $attention_dropout \
        --model_name MGAT_2 &

    let number=$number+1


done
wait

for attention_dropout in 0.2 0.4 0.6
do 

    python -u train.py \
        --number $number \
        --num_epoch 50 \
        --dim_latent 256 \
        --batch_size 2048 \
        --weight_decay 0.1 \
        --l_r 0.001 \
        --data_path dataset/coolcats2/ \
        --reg_parm 1e-5 \
        --neg_sample 10 \
        --PATH_weight_save model/coolcats_2/ \
        --loss_alpha 0.2 \
        --attention_dropout $attention_dropout \
        --model_name MGAT_2 &

    let number=$number+1


done
wait


for attention_dropout in 0.2 0.4 0.6
do 

    python -u train.py \
        --number $number \
        --num_epoch 50 \
        --dim_latent 512 \
        --batch_size 2048 \
        --weight_decay 0.1 \
        --l_r 0.001 \
        --data_path dataset/coolcats2/ \
        --reg_parm 1e-5 \
        --neg_sample 10 \
        --PATH_weight_save model/coolcats_2/ \
        --loss_alpha 0.2 \
        --attention_dropout $attention_dropout \
        --model_name MGAT_2 &

    let number=$number+1


done
wait
number=0

for attention_dropout in 0.2 0.4 0.6
do 

    python -u train.py \
        --number $number \
        --num_epoch 50 \
        --dim_latent 128 \
        --batch_size 2048 \
        --weight_decay 0.1 \
        --l_r 0.001 \
        --data_path dataset/coolcats3/ \
        --reg_parm 1e-5 \
        --neg_sample 10 \
        --PATH_weight_save model/coolcats_3/ \
        --loss_alpha 0.2 \
        --attention_dropout $attention_dropout \
        --model_name MGAT_2 &

    let number=$number+1


done
wait

for attention_dropout in 0.2 0.4 0.6
do 

    python -u train.py \
        --number $number \
        --num_epoch 50 \
        --dim_latent 256 \
        --batch_size 2048 \
        --weight_decay 0.1 \
        --l_r 0.001 \
        --data_path dataset/coolcats3/ \
        --reg_parm 1e-5 \
        --neg_sample 10 \
        --PATH_weight_save model/coolcats_3/ \
        --loss_alpha 0.2 \
        --attention_dropout $attention_dropout \
        --model_name MGAT_2 &

    let number=$number+1


done
wait


for attention_dropout in 0.2 0.4 0.6
do 

    python -u train.py \
        --number $number \
        --num_epoch 50 \
        --dim_latent 512 \
        --batch_size 2048 \
        --weight_decay 0.1 \
        --l_r 0.001 \
        --data_path dataset/coolcats3/ \
        --reg_parm 1e-5 \
        --neg_sample 10 \
        --PATH_weight_save model/coolcats_3/ \
        --loss_alpha 0.2 \
        --attention_dropout $attention_dropout \
        --model_name MGAT_2 &

    let number=$number+1


done
