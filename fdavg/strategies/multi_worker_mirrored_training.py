import tensorflow as tf
from fdavg.strategies.naive import naive_training_loop
from fdavg.strategies.linear import linear_training_loop
from fdavg.strategies.sketch import sketch_training_loop, AmsSketch
from math import sqrt


def prepare_multi_worker_mirrored_train(exper_info):

    if exper_info['slurm']:
        strategy = tf.distribute.MultiWorkerMirroredStrategy(exper_info['slurm_cluster'])
    else:
        strategy = tf.distribute.MultiWorkerMirroredStrategy()

    with strategy.scope():
        multi_worker_model = exper_info['build_and_compile_model_fn']()

        multi_worker_dataset = strategy.distribute_datasets_from_function(
            lambda input_context: exper_info['dataset_fn'](exper_info['global_batch_size'], input_context)
        )

    return strategy, multi_worker_model, multi_worker_dataset


def multi_worker_mirrored_train(exper_info):
    epoch_metrics = None

    strategy, multi_worker_model, multi_worker_dataset = prepare_multi_worker_mirrored_train(exper_info)

    if exper_info['strat_name'] == 'naive':
        epoch_metrics = naive_training_loop(
            strategy=strategy,
            multi_worker_model=multi_worker_model,
            multi_worker_dataset=multi_worker_dataset,
            num_epochs=exper_info['num_epochs'],
            num_steps_per_epoch=exper_info['num_steps_per_epoch'],
            theta=exper_info['theta'],
            per_replica_batch_size=exper_info['per_replica_batch_size']
        )

    if exper_info['strat_name'] == 'linear':
        epoch_metrics = linear_training_loop(
            strategy=strategy,
            multi_worker_model=multi_worker_model,
            multi_worker_dataset=multi_worker_dataset,
            num_epochs=exper_info['num_epochs'],
            num_steps_per_epoch=exper_info['num_steps_per_epoch'],
            theta=exper_info['theta'],
            per_replica_batch_size=exper_info['per_replica_batch_size']
        )

    if exper_info['strat_name'] == 'sketch':

        sketch_width, sketch_depth = 250, 5
        ams_sketch = AmsSketch(width=sketch_width, depth=sketch_depth, with_seed=True)
        epsilon = 1. / sqrt(sketch_width)

        epoch_metrics = sketch_training_loop(
            strategy=strategy,
            multi_worker_model=multi_worker_model,
            multi_worker_dataset=multi_worker_dataset,
            num_epochs=exper_info['num_epochs'],
            num_steps_per_epoch=exper_info['num_steps_per_epoch'],
            theta=exper_info['theta'],
            per_replica_batch_size=exper_info['per_replica_batch_size'],
            ams_sketch=ams_sketch,
            epsilon=epsilon
        )

    return epoch_metrics
