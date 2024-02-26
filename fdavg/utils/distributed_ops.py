import tensorflow as tf
from fdavg.models.miscellaneous import update_model_vars


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

        accuracy_metric = _multi_worker_model.metrics[1]

        accuracy_metric.update_state(labels, predictions)

    avg_train_model_vars = strategy.run(average_model_trainable_variables, args=(multi_worker_model,))

    # Update testing model's trainable variables per-replica
    update_model_vars(multi_worker_model_for_test.trainable_variables, avg_train_model_vars)

    if multi_worker_model.non_trainable_variables:
        avg_non_train_model_vars = strategy.run(average_model_non_trainable_variables, args=(multi_worker_model,))

        # Update testing model's trainable non-variables per-replica
        update_model_vars(multi_worker_model_for_test.non_trainable_variables, avg_non_train_model_vars)

    print(f"\n\n\n{multi_worker_model_for_test.metrics}\n\n\n\n")

    for inputs in distributed_test_dataset:
        strategy.run(test_step, args=(multi_worker_model_for_test, inputs))

    return multi_worker_model_for_test.metrics[1].result()
