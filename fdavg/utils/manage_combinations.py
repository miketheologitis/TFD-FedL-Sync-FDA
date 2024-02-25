import json
import os
from fdavg.models.models import (build_and_compile_densenet_for_cifar10, build_and_compile_lenet5_for_mnist,
                                 build_and_compile_advanced_cnn_for_mnist)
from functools import partial
from fdavg.data.preprocessing import mnist_dataset_fn, cifar10_dataset_fn, MNIST_N_TRAIN, CIFAR10_N_TRAIN
from fdavg.metrics.metrics import TestId

script_directory = os.path.dirname(os.path.abspath(__file__))
# Relative path to the tmp directory
tmp_dir = '../../metrics/tmp'
comb_dir = os.path.normpath(os.path.join(script_directory, f'{tmp_dir}/combinations'))


def get_experiment_information(comb_file_id, sim_id):
    with open(f'{comb_dir}/{comb_file_id}.json', 'r') as f:
        all_combinations = json.load(f)
    return all_combinations[sim_id]


def derive_extra_params(exper_info):

    exper_info['global_batch_size'] = (exper_info['per_replica_batch_size'] * exper_info['num_replicas'])
    exper_info['per_worker_batch_size'] = exper_info['per_replica_batch_size'] * exper_info['num_replicas_per_worker']
    exper_info['num_workers'] = int(exper_info['num_replicas'] / exper_info['num_replicas_per_worker'])

    if exper_info["ds_name"] == "MNIST":
        exper_info["num_steps_per_epoch"] = (MNIST_N_TRAIN // exper_info['num_of_workers'] //
                                             exper_info['per_worker_batch_size'])

        exper_info['dataset_fn'] = mnist_dataset_fn

        if exper_info["nn_name"] == 'LeNet-5':
            exper_info['build_and_compile_model_fn'] = build_and_compile_lenet5_for_mnist

        if exper_info["nn_name"] == 'AdvancedCNN':
            exper_info['build_and_compile_model_fn'] = build_and_compile_advanced_cnn_for_mnist

    if exper_info["ds_name"] == "CIFAR10":
        exper_info["num_steps_per_epoch"] = (CIFAR10_N_TRAIN // exper_info['num_of_workers'] //
                                             exper_info['per_worker_batch_size'])

        exper_info['dataset_fn'] = cifar10_dataset_fn

        # Assumed NN is a DenseNet
        exper_info['build_and_compile_model_fn'] = partial(
            build_and_compile_densenet_for_cifar10,
            exper_info["nn_name"]
        )


def get_test_id(exper_info):
    return TestId(
        exper_info['ds_name'], exper_info['strat_name'], exper_info['num_replicas'],
        exper_info['per_replica_batch_size'], exper_info['theta'], exper_info['nn_name']
    )