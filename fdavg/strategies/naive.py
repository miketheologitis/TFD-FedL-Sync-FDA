import time
import tensorflow as tf
from fdavg.strategies.fda import fda_step_fn
from fdavg.models.miscellaneous import trainable_vars_as_vector, update_distributed_model_vars_from_tensors
from fdavg.utils.distributed_ops import aggregate_models
from fdavg.metrics.metrics import EpochMetrics


def naive_rtc(multi_worker_model, w_t0, theta):
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

    avg_drift_sq = tf.distribute.get_replica_context().all_reduce(
        tf.distribute.ReduceOp.MEAN, drift_sq
    )

    return avg_drift_sq > theta


def naive_training_loop(strategy, multi_worker_model, multi_worker_dataset,
                        num_epochs, num_steps_per_epoch, theta, per_replica_batch_size):

    epoch_metrics = []

    epoch = 0

    w_t0 = trainable_vars_as_vector(multi_worker_model.trainable_variables)  # tf.Tensor vector w/ shape=(d,)

    while epoch <= num_epochs:
        start_epoch_time = time.time()

        iterator = iter(multi_worker_dataset)
        total_loss, num_epoch_rounds, num_epoch_steps, total_steps = 0.0, 0, 0, 0

        while num_epoch_steps <= num_steps_per_epoch:

            #loss = fda_train_step(strategy, iterator, multi_worker_model, per_replica_batch_size)
            #total_loss += loss
            strategy.run(fda_step_fn, args=(next(iterator), multi_worker_model, per_replica_batch_size))
            num_epoch_steps += 1
            total_steps += 1

            if naive_rtc(multi_worker_model, w_t0, theta):
                print(f"Synchronization Needed in Step {num_epoch_steps}")
                # Synchronization needed - Round terminates
                synced_model_vars = aggregate_models(multi_worker_model.trainable_variables)
                update_distributed_model_vars_from_tensors(multi_worker_model.trainable_variables, synced_model_vars)

                w_t0 = trainable_vars_as_vector(multi_worker_model.trainable_variables)
                num_epoch_rounds += 1

        # TODO: epoch ends, find accuracy
        #train_loss = total_loss / num_epoch_steps
        epoch += 1

        epoch_duration_sec = time.time() - start_epoch_time
        met = EpochMetrics(epoch, num_epoch_rounds, total_steps, epoch_duration_sec, 0.0)
        epoch_metrics.append(met)

        print(met)

    return epoch_metrics


