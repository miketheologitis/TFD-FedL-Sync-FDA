```shell
python -m fdavg.utils.create_combinations --comb_file_id 0 --ds_name MNIST --b 32 --e 2 --strat_name naive --nn LeNet-5 --th 2.0 --num_replicas 8 --walltime 00:07:00 --slurm
```

```shell
python -m submitter --comb_file_id 0
```