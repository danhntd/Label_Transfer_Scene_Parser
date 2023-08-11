export CUDA_VISIBLE_DEVICES=0
export NGPUS=1

# 3.3. Semantic Scene Parser Training
CUDA_VISIBLE_DEVICES=0 python train_val_CL_CE6FL4_stage1_cosine_UNIT.py \
        --dataset Cityscapes \
        --save_dir <path/to/save/result>/run_stage1_combine_CE6FL4/
        
# 3.4. Inference on Unlabeled Nighttime Data  
CUDA_VISIBLE_DEVICES=0 python infer.py \
    --experiment_dir ./Semantic_Segmentor/run_stage1_combine_CE6FL4 \
    --path_to_unlabel_set /path/to/unlabel/set/ \
    --path_to_save /path/to/dataset/Cityscapes/
    

# 3.5. Semantic Scene Parser Re-training 
CUDA_VISIBLE_DEVICES=0 python train_val_CL_CE6FL4_stage2_cosine_UNIT.py \
        --dataset Cityscapes \
        --save_dir <path/to/save/result>/run_stage2_combine_CE6FL4/
        --checkpoint ./Semantic_Segmentor/saved_checkpoints/run_stage1_combine_CE6FL4/Cityscapes/fpn-resnet101/model_best.pth.tar
   
# For testing on the trained models:    
CUDA_VISIBLE_DEVICES=0 python test.py \
        --dataset Cityscapes \ 
        --experiment_dir ./Semantic_Segmentor/run_stage2_combine_CE6FL4

# For inferencing on testing images:
CUDA_VISIBLE_DEVICES=0 python predict.py \
    --experiment_dir ./Semantic_Segmentor/run_stage2_combine_CE6FL4
    --path_to_save /path/to/destination/folder/
    --path_to_test_set /path/to/dataset/Cityscapes/


