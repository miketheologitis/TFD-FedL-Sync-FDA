import time
import tensorflow as tf
from fdavg.strategies.fda import fda_step_fn
from fdavg.models.miscellaneous import trainable_vars_as_vector
from fdavg.metrics.metrics import EpochMetrics
from fdavg.utils.distributed_ops import average_and_sync_model_trainable_variables


def ksi_unit(w_t0, w_tminus1):
    """
    Calculates the heuristic unit vector ksi.

    Args:
    - w_t0 (tf.Tensor): Initial model parameters for the current round. Shape=(d,).
    - w_tminus1 (tf.Tensor): Model parameters from the previous round. Shape=(d,).

    Returns:
    - tf.Tensor: The heuristic unit vector ksi.
    """
    if tf.reduce_all(tf.equal(w_t0, w_tminus1)):
        # if equal then ksi becomes a random vector (will only happen in round 1)
        ksi = tf.random.stateless_normal(shape=w_t0.shape, seed=[1, 2])

    else:
        ksi = w_t0 - w_tminus1

    # Normalize and return
    return tf.divide(ksi, tf.norm(ksi))


def linear_var_approx(multi_worker_model, w_t0, w_tminus1):
    """
    Round Terminating Condition for NaiveFDA

    This function computes the drift of the model's trainable variables from a baseline state, averages the squared
    drift across all replicas, and checks if this average exceeds a specified threshold. The operation ensures a
    unified decision across all replicas, returning a single boolean value that is consistent across the distributed
    context

    Args:
        multi_worker_model (tf.keras.Model): The distributed model being trained.
        w_t0 (tf.Tensor): The last round's model.
        w_tminus1 (tf.Tensor): The last, last round's model.
        theta (float): The variance threshold.

    Returns:
        bool: A single boolean value, identical across all replicas, indicating whether round should terminate (True)
        or not (False).
    """

    drift = trainable_vars_as_vector(multi_worker_model.trainable_variables) - w_t0

    # ||D(t)_i||^2 , shape = ()
    drift_sq = tf.reduce_sum(tf.square(drift))

    ksi = ksi_unit(w_t0, w_tminus1)
    # ksi * Delta_i (* is dot) , shape = ()
    ksi_dot_drift = tf.reduce_sum(tf.multiply(ksi, drift))

    avg_drift_sq, avg_ksi_dot_drift = tf.distribute.get_replica_context().all_reduce(
        tf.distribute.ReduceOp.MEAN, [drift_sq, ksi_dot_drift]
    )

    return avg_drift_sq - avg_ksi_dot_drift**2


def linear_training_loop(strategy, multi_worker_model, multi_worker_dataset,
                         num_epochs, num_steps_per_epoch, theta, per_replica_batch_size):

    epoch_metrics = []

    w_t0 = trainable_vars_as_vector(multi_worker_model.trainable_variables)  # tf.Tensor vector w/ shape=(d,)
    w_tminus1 = w_t0  # tf.Tensor vector w/ shape=(d,)

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
            est_var = strategy.run(linear_var_approx, args=(multi_worker_model, w_t0, w_tminus1))

            if est_var > theta:
                # All-reduce w/ averaging and synchronization of all per-replica models. After this all per-replica
                # models are the same (invokes `average_and_sync_model_trainable_variables` on each replica).
                strategy.run(average_and_sync_model_trainable_variables, args=(multi_worker_model,))

                w_tminus1 = w_t0
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
