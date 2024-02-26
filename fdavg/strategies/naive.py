import time
import tensorflow as tf
from fdavg.strategies.fda import fda_step_fn
from fdavg.models.miscellaneous import trainable_vars_as_vector, update_distributed_model_vars_from_tensors
from fdavg.utils.distributed_ops import aggregate_models
from fdavg.metrics.metrics import EpochMetrics


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


def aggr_models(multi_worker_model):

    return tf.distribute.get_replica_context().all_reduce(
        tf.distribute.ReduceOp.MEAN, multi_worker_model.trainable_variables
    )


def naive_training_loop(strategy, multi_worker_model, multi_worker_dataset,
                        num_epochs, num_steps_per_epoch, theta, per_replica_batch_size):

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
                # Synchronization needed - Round terminates
                #TODO: synced_model_vars = aggregate_models(multi_worker_model.trainable_variables)

                print("hi 1")
                synced_model_vars = strategy.run(aggr_models, args=(multi_worker_model,))
                tmp = trainable_vars_as_vector(synced_model_vars)
                print(f"tmp: {tf.reduce_mean(tmp)}")

                #tmp = trainable_vars_as_vector(synced_model_vars)
                #print(f"Sync: {num_total_rounds} here ----> {tf.reduce_mean(tmp)}")

                #TODO: update_distributed_model_vars_from_tensors(multi_worker_model.trainable_variables, synced_model_vars)

                w_t0 = trainable_vars_as_vector(multi_worker_model.trainable_variables)
                num_total_rounds += 1

        # TODO: epoch ends, find accuracy
        epoch += 1

        # ---- METRICS ----
        epoch_duration_sec = time.time() - start_epoch_time
        e_met = EpochMetrics(epoch, num_total_rounds, num_total_steps, epoch_duration_sec, 0.0)
        epoch_metrics.append(e_met)
        print(e_met)
        # ---- METRICS ----

    return epoch_metrics


