python main.py \
--model_type xlnet \
--model_name_or_path xlnet-base-cased \
--task_name Wiki500k \
--do_eval \
--eval_all_checkpoints \
--overwrite_output_dir \
--data_dir ../data/Wiki500k \
--max_seq_length 128 \
--per_gpu_train_batch_size=16 \
--per_gpu_eval_batch_size=8 \
--learning_rate_x 5e-5 \
--learning_rate_h 1e-4 \
--learning_rate_a 2e-3 \
--num_train_epochs 15.0 \
--output_dir ../models/Wiki500k \
--pos_label 274 \
--adaptive_cutoff 125252 250504 375756 \
--div_value 2 \
--logging_steps 15000 \
--save_steps 15000 \
--gpu 0,1





