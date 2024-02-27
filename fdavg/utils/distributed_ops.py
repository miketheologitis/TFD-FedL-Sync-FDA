import tensorflow as tf
from fdavg.models.miscellaneous import update_model_vars


def average_model_trainable_variables(multi_worker_model):

    return tf.distribute.get_replica_context().all_reduce(
        tf.distribute.ReduceOp.MEAN, multi_worker_model.trainable_variables
    )


def average_and_sync_model_trainable_variables(multi_worker_model):

    avg_train_model_vars = average_model_trainable_variables(multi_worker_model)

    update_model_vars(multi_worker_model.trainable_variables, avg_train_model_vars)


def accuracy_of_distributed_model(strategy, multi_worker_model, multi_worker_model_for_test, test_accuracy_metric,
                                  distributed_test_dataset):

    test_accuracy_metric.reset_states()

    def test_step(_multi_worker_model, _inputs, _test_accuracy_metric):
        images, labels = _inputs

        predictions = _multi_worker_model(images, training=False)

        _test_accuracy_metric.update_state(labels, predictions)

    avg_train_model_vars = strategy.run(average_model_trainable_variables, args=(multi_worker_model,))

    # Update testing model's trainable variables per-replica
    update_model_vars(multi_worker_model_for_test.trainable_variables, avg_train_model_vars)

    # Non-Trainable variables are kept in-sync by tensorflow distributed behind the scenes. No all-reduce needed.
    if multi_worker_model.non_trainable_variables:
        # Update testing model's trainable non-variables per-replica
        update_model_vars(
            multi_worker_model_for_test.non_trainable_variables, multi_worker_model.non_trainable_variables
        )

    for inputs in distributed_test_dataset:
        strategy.run(test_step, args=(multi_worker_model_for_test, inputs, test_accuracy_metric))

    return test_accuracy_metric.result().numpy()


def acc_testing_purposes(strategy, multi_worker_model):
    """ Do not use. Testing purposes """

    from fdavg.models.models import build_and_compile_advanced_cnn_for_mnist
    from fdavg.data.preprocessing import mnist_load_data

    tmp = build_and_compile_advanced_cnn_for_mnist()
    _, _, X, y = mnist_load_data()

    avg_train_model_vars = strategy.run(average_model_trainable_variables, args=(multi_worker_model,))

    # Update testing model's trainable variables per-replica
    update_model_vars(tmp.trainable_variables, avg_train_model_vars)

    _, acc = tmp.evaluate(x=X, y=y, batch_size=256, verbose=0)

    return acc
