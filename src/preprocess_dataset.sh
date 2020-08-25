# python3 ./keyword/data/kp20k_dataset.py --source_dataset ./keyword/rsc/preprocessed/kp20k.train.256.16.json --output_path ./keyword/rsc/dataset/kp20k.train.dataset.256.16.pt --max_src_seq_len 256 --max_trg_seq_len 16

python3 ./keyword/data/kp20k_dataset.py --source_dataset ./keyword/rsc/preprocessed/kp20k.test.256.16.json --output_path ./keyword/rsc/dataset/kp20k.test.dataset.256.16.pt --max_src_seq_len 256 --max_trg_seq_len 16

python3 ./keyword/data/kp20k_dataset.py --source_dataset ./keyword/rsc/preprocessed/kp20k.valid.256.16.json --output_path ./keyword/rsc/dataset/kp20k.valid.dataset.256.16.pt --max_src_seq_len 256 --max_trg_seq_len 16
