import time
import tensorflow as tf
from fdavg.strategies.fda import fda_step_fn
from fdavg.models.miscellaneous import trainable_vars_as_vector
from fdavg.metrics.metrics import EpochMetrics
from fdavg.utils.distributed_ops import average_and_sync_model_trainable_variables, accuracy_of_distributed_model, acc_test


def naive_var_approx(multi_worker_model, w_t0):
    """
    Round Terminating Condition for NaiveFDA

    This function computes the drift of the model's trainable variables from a baseline state, averages the squared
    drift across all replicas, and checks if this average exceeds a specified threshold. The operation ensures a
    unified decision across all replicas, returning a single boolean value that is consistent across the distributed
    context

    Args:
        multi_worker_model (tf.keras.Model): The distributed model being trained.
        w_t0 (tf.Tensor): The last round's model.
        theta (float): The variance threshold.

    Returns:
        bool: A single boolean value, identical across all replicas, indicating whether round should terminate (True)
        or not (False).
    """

    drift = trainable_vars_as_vector(multi_worker_model.trainable_variables) - w_t0

    drift_sq = tf.reduce_sum(tf.square(drift))

    # Per-replica all-reduce
    avg_drift_sq = tf.distribute.get_replica_context().all_reduce(
        tf.distribute.ReduceOp.MEAN, drift_sq
    )

    return avg_drift_sq


def naive_training_loop(strategy, multi_worker_model, multi_worker_dataset, multi_worker_model_for_test,
                        multi_worker_test_dataset, test_accuracy_metric, num_epochs, num_steps_per_epoch, theta,
                        per_replica_batch_size):

    epoch_metrics = []

    w_t0 = trainable_vars_as_vector(multi_worker_model.trainable_variables)  # tf.Tensor vector w/ shape=(d,)

    epoch, num_total_rounds, num_total_steps = 0, 0, 0

    while epoch <= num_epochs:
        start_epoch_time = time.time()

        iterator = iter(multi_worker_dataset)
        num_epoch_steps = 0

        while num_epoch_steps <= num_steps_per_epoch:

            # Train Step
            strategy.run(fda_step_fn, args=(next(iterator), multi_worker_model, per_replica_batch_size))
            num_epoch_steps += 1
            num_total_steps += 1

            # Estimate variance, invokes `naive_var_approx` on each replica. After all-reduce operation `est_var`
            # is the same for all replicas, managed by each worker (who is responsible for some replicas).
            est_var = strategy.run(naive_var_approx, args=(multi_worker_model, w_t0))

            if est_var > theta:
                # All-reduce w/ averaging and synchronization of all per-replica models. After this all per-replica
                # models are the same (invokes `average_and_sync_model_trainable_variables` on each replica).
                strategy.run(average_and_sync_model_trainable_variables, args=(multi_worker_model,))

                w_t0 = trainable_vars_as_vector(multi_worker_model.trainable_variables)
                num_total_rounds += 1

        # TODO: epoch ends, find accuracy
        epoch += 1

        # ---- METRICS ----
        epoch_duration_sec = time.time() - start_epoch_time
        acc = accuracy_of_distributed_model(
            strategy, multi_worker_model, multi_worker_model_for_test, test_accuracy_metric, multi_worker_test_dataset
        )
        e_met = EpochMetrics(epoch, num_total_rounds, num_total_steps, epoch_duration_sec, acc)
        epoch_metrics.append(e_met)
        print(e_met)
        test_acc = acc_test(strategy, multi_worker_model)
        print(f"Found this acc: {test_acc}")
        # ---- METRICS ----

    return epoch_metrics


