CUDA_VISIBLE_DEVICES=0
export NGPUS=1
cd Label_Transfer_Scene_Parser/Domain_Translator

python test_batch.py \
                --trainer UNIT \
                --config /path_to_unit_day2night_folder_add_vgg_loss.yaml \
                --input_folder path_to_folder_testA/ \
                --output_folder /output_testA/  \
                --checkpoint /path_to_ckpt_day2night_gen_00330000.pt \
                --a2b 1 \
                --output_only