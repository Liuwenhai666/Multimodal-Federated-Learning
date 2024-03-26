for fed_alg in fed_avg fed_opt; do
    taskset -c 1-60 python3 train.py --hid_size 128 --sample_rate 0.1 --learning_rate 0.05 --num_epochs 300 --en_att --att_name fuse_base --fed_alg $fed_alg --global_learning_rate 0.025 --mu 0.01
    # taskset 50 python3 train.py --hid_size 128 --sample_rate 0.1 --learning_rate 0.05 --num_epochs 200 --fed_alg $fed_alg --global_learning_rate 0.025 --mu 0.01
done
