python -m torch.distributed.launch \
--nproc_per_node=8 \
--use_env \
main.py \
        --output_dir logs/vcoco \
        \
        \
        --dataset_file vcoco \
        --hoi_path data/v-coco \
        --num_obj_classes 81 \
        --num_verb_classes 29 \
        \
        --dec_layers_hopd 3 \
        --dec_layers_interaction 3 \
        \
        --epochs 90 \
        --lr_drop 60 \
        --num_queries 100 \
        \
        \
        --wandb\
        --backbone resnet50 \
        --pretrained params/detr-r50-pre-2stage-q100-vcoco.pth
        #--use_nms_filter


#finetune
: '
python -m torch.distributed.launch \
        --nproc_per_node=8 \
        --use_env \
        main.py \
        --pretrained logs/checkpoint_last.pth \
        --output_dir logs/ \
        --dataset_file vcoco \
        --hoi_path data/v-coco \
        --num_obj_classes 81 \
        --num_verb_classes 29 \
        --backbone resnet50 \
        --num_queries 100 \
        --dec_layers_hopd 3 \
        --dec_layers_interaction 3 \
        --epochs 10 \
        --freeze_mode 1 \
        --verb_reweight \
        --lr 1e-5 \
        --lr_backbone 1e-6 \
        --use_nms_filter
'

python generate_vcoco_official.py \
        --param_path logs/vcoco/checkpoint_last.pth \
        --save_path logs/vcoco/vcoco.pickle \
        --hoi_path data/v-coco \
        --dec_layers_hopd 3 \
        --dec_layers_interaction 3
       # --use_nms_filter

python data/v-coco/vsrl_eval.py logs/vcoco/vcoco.pickle