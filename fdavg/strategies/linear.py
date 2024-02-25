import time
import tensorflow as tf
from fdavg.strategies.fda import fda_train_step
from fdavg.models.miscellaneous import trainable_vars_as_vector, update_distributed_model_vars_from_tensors
from fdavg.utils.distributed_ops import aggregate_models


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


def linear_rtc(multi_worker_model, w_t0, w_tminus1, theta):
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

    return avg_drift_sq - avg_ksi_dot_drift**2 > theta


def linear_training_loop(strategy, multi_worker_model, multi_worker_dataset,
                         num_epochs, num_steps_per_epoch, theta, per_replica_batch_size):

    step_metrics, epoch_metrics = [], []

    epoch = 0

    w_t0 = trainable_vars_as_vector(multi_worker_model.trainable_variables)  # tf.Tensor vector w/ shape=(d,)
    w_tminus1 = w_t0  # tf.Tensor vector w/ shape=(d,)

    while epoch <= num_epochs:
        iterator = iter(multi_worker_dataset)
        total_loss, num_epoch_rounds, num_epoch_steps = 0.0, 0, 0

        while num_epoch_steps <= num_steps_per_epoch:
            loss = fda_train_step(strategy, iterator, multi_worker_model, per_replica_batch_size)
            total_loss += loss
            num_epoch_steps += 1

            if linear_rtc(multi_worker_model, w_t0, w_tminus1, theta):
                # Synchronization needed - Round terminates
                synced_model_vars = aggregate_models(multi_worker_model.trainable_variables)
                update_distributed_model_vars_from_tensors(multi_worker_model.trainable_variables, synced_model_vars)

                w_tminus1 = w_t0
                w_t0 = trainable_vars_as_vector(multi_worker_model.trainable_variables)
                num_epoch_rounds += 1

        # TODO: epoch ends, find accuracy

        train_loss = total_loss / num_epoch_steps
        epoch += 1

        print(f"Epoch: {epoch}, train_loss:{train_loss}")

    return step_metrics, epoch_metrics
