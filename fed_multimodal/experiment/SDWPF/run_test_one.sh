alpha=0.1
fuse_base="fed_opt"
taskset -c 1-30 python3 train.py --alpha 5.0 --sample_rate 0.1 --learning_rate 0.05 --global_learning_rate 0.025 --num_epochs 200 --fed_alg fed_avg --mu 0.01 --en_att --att_name fuse_base --hid_size 128