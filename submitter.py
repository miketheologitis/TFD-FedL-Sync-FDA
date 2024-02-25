import subprocess
import argparse
import os
import json

ARIS_NODE_MEM = '55G'

slurm_template = """#!/bin/bash -l

####################################
#     ARIS slurm script template   #
#                                  #
# Submit script: sbatch filename   #
#                                  #
####################################

#SBATCH --job-name={job_name}    # Job name
#SBATCH --output=metrics/tmp/slurm_out/{job_name}.out   # Stdout %j expands to jobId, %a is array task index
#SBATCH --error=metrics/tmp/slurm_out/{job_name}.err   # Stderr %j expands to jobId, %a is array task index
#SBATCH --ntasks={n_tasks} # Number of tasks(processes)
#SBATCH --nodes={n_nodes}     # Number of nodes requested
#SBATCH --ntasks-per-node=1     # Tasks per node
#SBATCH --mem={mem}     # memory per NODE
#SBATCH --time={walltime}   # walltime
#SBATCH --partition=gpu    # Partition
#SBATCH --account=pa240202

## LOAD MODULES ##
module purge            # clean up loaded modules
module load gnu/8
module load cuda/10.1.168
module load intel/18
module load intelmpi/2018
module load python/3.8.13
module load tftorch/270-191

echo node name, name of the current node: ${SLURMD_NODENAME}
echo job node list, list of all nodes allocated to the job: ${SLURM_JOB_NODELIST}
echo job num nodes, number of nodes the job is using: ${SLURM_JOB_NUM_NODES}
echo step node list: ${SLURM_STEP_NODELIST}
echo "Running program"

export TF_XLA_FLAGS="--tf_xla_enable_xla_devices"

srun python -u -m fdavg.main --comb_file_id {comb_id} --exper_id {exper_id}

echo "Finished program"
"""

script_directory = os.path.dirname(os.path.abspath(__file__))
# Relative path to the tmp directory
tmp_dir = 'metrics/tmp'
comb_dir = os.path.normpath(os.path.join(script_directory, f'{tmp_dir}/combinations'))


def get_all_experiments_information(comb_file_id):
    with open(f'{comb_dir}/{comb_file_id}.json', 'r') as file:
        all_combinations = json.load(file)
    return all_combinations


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--comb_file_id', type=int, help="The combinations prefix, i.e., <PREFIX>.json.",
                        required=True)
    args = parser.parse_args()

    experiments = get_all_experiments_information(args.comb_file_id)

    num_of_submits = len(experiments)

    for i, exper in enumerate(experiments):

        num_workers = int(exper['num_replicas'] / exper['num_replicas_per_worker'])

        slurm_script = slurm_template.format(
            job_name=f"c{args.comb_file_id}_s{i}",
            n_tasks=num_workers,
            n_nodes=num_workers,
            mem=ARIS_NODE_MEM,
            walltime=exper['walltime'],
            comb_id=args.comb_file_id,
            exper_id=i
        )

        # Save the Slurm script content to a temporary file
        with open("tmp_slurm_script.slurm", "w") as f:
            f.write(slurm_script)

        # Submit the Slurm script using sbatch
        result = subprocess.run(["sbatch", "tmp_slurm_script.slurm"], capture_output=True, text=True)
        print(result)
        print(f"Submitted slurm job {i + 1}/{num_of_submits} for combination file {args.comb_file_id}.json.")
        print()

    # Remove the temporary file
    os.remove("tmp_slurm_script.slurm")
