: '
python -m torch.distributed.launch \
--nproc_per_node=8 \
--use_env \
'

torchrun --nproc_per_node=8 \
main.py \
        --output_dir logs/hico \
        \
        \
        --num_workers 0 \
        --dataset_file hico \
        --hoi_path data/hico_20160224_det \
        --num_obj_classes 80 \
        --num_verb_classes 117 \
        \
        --dec_layers_hopd 3 \
        --dec_layers_interaction 3 \
        \
        \
        --num_queries 64 \
        --pretrained params/detr-r50-pre-2stage-q64.pth \
        --lr_drop 60 \
        --epochs 90 \
        \
        --wandb\
        --backbone resnet50
        #--use_nms_filter



#finetune
: '
python -m torch.distributed.launch \
--nproc_per_node=8 \
--use_env \
main.py \
        --output_dir logs/hico \
        --dataset_file hico \
        --hoi_path data/hico_20160224_det \
        --num_obj_classes 80 \
        --num_verb_classes 117 \
        \
        --num_queries 64 \
        --dec_layers_hopd 3 \
        --dec_layers_interaction 3 \
        \
        \
        --freeze_mode 1 \
        --  \
        --verb_reweight \
        --use_nms_filter\
        \
        --epochs 10 \
        --lr 1e-5 \
        --lr_backbone 1e-6 \
        --backbone resnet50 \
        --pretrained logs/hico/checkpoint_last.pth
'
