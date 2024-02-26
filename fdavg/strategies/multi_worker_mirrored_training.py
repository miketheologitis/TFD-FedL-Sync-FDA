import tensorflow as tf
from fdavg.strategies.naive import naive_training_loop
from fdavg.strategies.linear import linear_training_loop
from fdavg.strategies.sketch import sketch_training_loop, AmsSketch
from fdavg.models.miscellaneous import update_model_vars
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

        # TODO: testing dataset, tmp model for testing
        multi_worker_model_for_test = exper_info['build_and_compile_model_fn']()

        multi_worker_test_dataset = strategy.distribute_datasets_from_function(
            lambda input_context: exper_info['test_dataset_fn'](exper_info['global_batch_size'], input_context)
        )

    return strategy, multi_worker_model, multi_worker_dataset, multi_worker_model_for_test, multi_worker_test_dataset


def multi_worker_mirrored_train(exper_info):
    epoch_metrics = None

    strategy, multi_worker_model, multi_worker_dataset, multi_worker_model_for_test, multi_worker_test_dataset = (
        prepare_multi_worker_mirrored_train(exper_info))

    if exper_info['strat_name'] == 'naive':
        epoch_metrics = naive_training_loop(
            strategy=strategy,
            multi_worker_model=multi_worker_model,
            multi_worker_dataset=multi_worker_dataset,
            multi_worker_model_for_test=multi_worker_model_for_test,
            multi_worker_test_dataset=multi_worker_test_dataset,
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
            multi_worker_model_for_test=multi_worker_model_for_test,
            multi_worker_test_dataset=multi_worker_test_dataset,
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
            multi_worker_model_for_test=multi_worker_model_for_test,
            multi_worker_test_dataset=multi_worker_test_dataset,
            num_epochs=exper_info['num_epochs'],
            num_steps_per_epoch=exper_info['num_steps_per_epoch'],
            theta=exper_info['theta'],
            per_replica_batch_size=exper_info['per_replica_batch_size'],
            ams_sketch=ams_sketch,
            epsilon=epsilon
        )

    return epoch_metrics


def average_model_trainable_variables(multi_worker_model):

    return tf.distribute.get_replica_context().all_reduce(
        tf.distribute.ReduceOp.MEAN, multi_worker_model.trainable_variables
    )


def average_model_non_trainable_variables(multi_worker_model):

    return tf.distribute.get_replica_context().all_reduce(
        tf.distribute.ReduceOp.MEAN, multi_worker_model.non_trainable_variables
    )


def average_and_sync_model_trainable_variables(multi_worker_model):

    avg_train_model_vars = average_model_trainable_variables(multi_worker_model)

    update_model_vars(multi_worker_model.trainable_variables, avg_train_model_vars)


def accuracy_of_distributed_model(strategy, multi_worker_model, multi_worker_model_for_test, distributed_test_dataset):

    def test_step(_multi_worker_model, _inputs):
        images, labels = _inputs

        predictions = _multi_worker_model(images, training=False)

        accuracy_metric = _multi_worker_model.metrics[0]

        accuracy_metric.update_state(labels, predictions)

    avg_train_model_vars = strategy.run(average_model_trainable_variables, args=(multi_worker_model,))

    # Update testing model's trainable variables per-replica
    update_model_vars(multi_worker_model_for_test.trainable_variables, avg_train_model_vars)

    if multi_worker_model.non_trainable_variables:
        avg_non_train_model_vars = strategy.run(average_model_non_trainable_variables, args=(multi_worker_model,))

        # Update testing model's trainable non-variables per-replica
        update_model_vars(multi_worker_model_for_test.non_trainable_variables, avg_non_train_model_vars)

    for inputs in distributed_test_dataset:
        strategy.run(test_step, args=(multi_worker_model_for_test, inputs))

    return multi_worker_model_for_test.metrics[0].result()
