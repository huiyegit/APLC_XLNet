python main.py \
--model_type xlnet \
--model_name_or_path xlnet-base-cased \
--task_name Wiki10 \
--do_eval \
--eval_all_checkpoints \
--overwrite_output_dir \
--data_dir ../data/Wiki10 \
--max_seq_length 512 \
--per_gpu_train_batch_size=6 \
--per_gpu_eval_batch_size=12 \
--learning_rate_x 1e-5 \
--learning_rate_h 1e-4 \
--learning_rate_a 1e-3 \
--num_train_epochs 6.0 \
--output_dir ../models/Wiki10/ \
--pos_label 30 \
--adaptive_cutoff 15469 \
--div_value 2 \
--logging_steps 500 \
--save_steps 500 \
--gpu 0,1





