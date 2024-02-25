import itertools
import json
import os
import argparse

# Get the directory containing the script
script_directory = os.path.dirname(os.path.abspath(__file__))

# Relative path to the tmp directory
tmp_dir = '../../metrics/tmp'

# Define the directories relative to the script's directory
directories = [
    os.path.normpath(os.path.join(script_directory, f'{tmp_dir}/slurm_out')),
    os.path.normpath(os.path.join(script_directory, f'{tmp_dir}/local_out')),
    os.path.normpath(os.path.join(script_directory, f'{tmp_dir}/epoch_metrics')),
    os.path.normpath(os.path.join(script_directory, f'{tmp_dir}/step_metrics')),
    os.path.normpath(os.path.join(script_directory, f'{tmp_dir}/combinations'))
]
# Create directories if they don't exist
for directory in directories:
    if not os.path.exists(directory):
        os.makedirs(directory)


def create_combinations(args):
    # Define the parameter values
    params = {
        "ds_name": args.ds_name,
        "nn_name": args.nn,
        "strat_name": args.fda,
        "per_replica_batch_size": args.b,
        "theta": args.th,
        "num_replicas": args.num_replicas
    }

    combinations = [
        dict(zip(params.keys(), values))
        for values in itertools.product(*params.values())
    ]

    for i, combination in enumerate(combinations):
        combination["num_epochs"] = args.e
        combination["slurm"] = args.slurm
        combination["walltime"] = args.walltime
        combination["workers_ip_port"] = args.workers_ip_port

        combination["exper_filename"] = (
            f"{combination['nn_name'].replace('-', '')}_{combination['strat_name']}_"
            f"b{combination['per_replica_batch_size']}"
            f"_c{combination['num_replicas']}_t{str(combination['theta']).replace('.', '')}"
        )

    if not args.append_to:
        with open(f'{os.path.join(script_directory, f"{tmp_dir}/combinations")}/{args.comb_file_id}.json', 'w') as f:
            json.dump(combinations, f)
            print(f"OK! Created {len(combinations)} combinations in {args.comb_file_id}.json, "
                  f"i.e., `n_sims` = {len(combinations)}.")
    else:
        with open(f'{os.path.join(script_directory, f"{tmp_dir}/combinations")}/{args.comb_file_id}.json', 'r') as f:
            old_combinations = json.load(f)
        old_combinations.extend(combinations)
        with open(f'{os.path.join(script_directory, f"{tmp_dir}/combinations")}/{args.comb_file_id}.json', 'w') as f:
            json.dump(old_combinations, f)
            print(f"OK! Appended {len(combinations)} combinations in {args.comb_file_id}.json,"
                  f" i.e., `n_sims` = {len(old_combinations)}.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--comb_file_id', type=int, help="The combinations prefix, i.e., <PREFIX>.json", required=True)
    parser.add_argument('--ds_name', nargs='+', type=str, help="The dataset name.", default=["MNIST"])
    parser.add_argument('--b', nargs='+', type=int, help="The batch size(s).")
    parser.add_argument('--e', type=int, help="Number of epochs.", required=True)
    parser.add_argument('--strat_name', nargs='+', type=str, help="The Strat name(s).", required=True)
    parser.add_argument('--nn', nargs='+', type=str,
                        help="The CNN name(s) ('LeNet-5' , 'AdvancedCNN', 'DenseNet121', 'DenseNet201')", required=True)
    parser.add_argument('--th', nargs='+', type=float, help="Theta threshold(s).", required=True)
    parser.add_argument('--num_replicas_per_worker', type=int, default=2,
                        help="Number of replicas (GPUs) per worker.")
    parser.add_argument('--num_replicas', nargs='+', type=int, help="Number of clients.",
                        default=[4, 8, 12, 16, 20])
    parser.add_argument('--workers_ip_port', default=[], nargs='+', type=str,
                        help='Workers IP1:PORT1 IP2:PORT2, for local simulation.')
    parser.add_argument('--walltime', type=str, help="The wall-time for slurm.", default="24:00:00")
    parser.add_argument('--append_to', action='store_true',
                        help="If given, then we append to the comb file.")
    parser.add_argument('--slurm', action='store_true', help="If given we run on slurm, else locally.")

    create_combinations(parser.parse_args())
