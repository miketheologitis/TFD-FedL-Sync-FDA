import time
import tensorflow as tf
import numpy as np
from fdavg.strategies.fda import fda_step_fn
from fdavg.models.miscellaneous import trainable_vars_as_vector, update_distributed_model_vars_from_tensors
from fdavg.utils.distributed_ops import aggregate_models
from fdavg.metrics.metrics import EpochMetrics


class AmsSketch:
    """
    AMS Sketch class for approximate second moment estimation.
    """

    def __init__(self, depth=5, width=250, with_seed=False):
        self.depth = tf.constant(depth)
        self.width = tf.constant(width)

        if with_seed:
            self.F = tf.random.stateless_uniform(shape=(6, depth), minval=0, maxval=(1 << 31) - 1, dtype=tf.int32,
                                                 seed=(1, 2))
        else:
            self.F = tf.random.uniform(shape=(6, depth), minval=0, maxval=(1 << 31) - 1, dtype=tf.int32)

        self.zeros_sketch = tf.zeros(shape=(self.depth, self.width), dtype=tf.float32)

        self.precomputed_dict = {}

    def precompute(self, d):
        pos_tensor = self.tensor_hash31(tf.range(d), self.F[0], self.F[1]) % self.width  # shape=(d, 5)

        self.precomputed_dict[('four', d)] = tf.cast(self.tensor_fourwise(tf.range(d)),
                                                     dtype=tf.float32)  # shape=(d, 5)

        range_tensor = tf.range(self.depth)  # shape=(5,)

        # Expand dimensions to create a 2D tensor with shape (1, `self.depth`)
        range_tensor_expanded = tf.expand_dims(range_tensor, 0)  # shape=(1, 5)

        # Use tf.tile to repeat the range `d` times
        repeated_range_tensor = tf.tile(range_tensor_expanded, [d, 1])  # shape=(d, 5)

        # shape=(`d`, `self.depth`, 2)
        self.precomputed_dict[('indices', d)] = tf.stack([repeated_range_tensor, pos_tensor],
                                                         axis=-1)  # shape=(d, 5, 2)

    @staticmethod
    def hash31(x, a, b):
        r = a * x + b
        fold = tf.bitwise.bitwise_xor(tf.bitwise.right_shift(r, 31), r)
        return tf.bitwise.bitwise_and(fold, 2147483647)

    @staticmethod
    def tensor_hash31(x, a, b):  # GOOD
        """ Assumed that x is tensor shaped (d,) , i.e., a vector (for example, indices, i.e., tf.range(d)) """

        # Reshape x to have an extra dimension, resulting in a shape of (k, 1)
        x_reshaped = tf.expand_dims(x, axis=-1)

        # shape=(`v_dim`, 7)
        r = tf.multiply(a, x_reshaped) + b

        fold = tf.bitwise.bitwise_xor(tf.bitwise.right_shift(r, 31), r)

        return tf.bitwise.bitwise_and(fold, 2147483647)

    def tensor_fourwise(self, x):
        """ Assumed that x is tensor shaped (d,) , i.e., a vector (for example, indices, i.e., tf.range(d)) """

        # 1st use the tensor hash31
        in1 = self.tensor_hash31(x, self.F[2], self.F[3])  # shape = (`x_dim`,  `self.depth`)

        # 2st use the tensor hash31
        in2 = self.tensor_hash31(x, in1, self.F[4])  # shape = (`x_dim`,  `self.depth`)

        # 3rd use the tensor hash31
        in3 = self.tensor_hash31(x, in2, self.F[5])  # shape = (`x_dim`,  `self.depth`)

        in4 = tf.bitwise.bitwise_and(in3, 32768)  # shape = (`x_dim`,  `self.depth`)

        return 2 * (tf.bitwise.right_shift(in4, 15)) - 1  # shape = (`x_dim`,  `self.depth`)

    def fourwise(self, x):
        result = 2 * (tf.bitwise.right_shift(tf.bitwise.bitwise_and(
            self.hash31(self.hash31(self.hash31(x, self.F[2], self.F[3]), x, self.F[4]), x, self.F[5]), 32768), 15)) - 1
        return result

    def sketch_for_vector(self, v):
        """ Extremely efficient computation of sketch with only using tensors.

        Args:
        - v (tf.Tensor): Vector to sketch. Shape=(d,).

        Returns:
        - tf.Tensor: An AMS - Sketch. Shape=(`depth`, `width`).
        """

        d = v.shape[0]

        if ('four', d) not in self.precomputed_dict:
            self.precompute(d)

        return self._sketch_for_vector(v, self.precomputed_dict[('four', d)], self.precomputed_dict[('indices', d)])

    @tf.function
    def _sketch_for_vector(self, v, four, indices):
        v_expand = tf.expand_dims(v, axis=-1)  # shape=(d, 1)

        # shape=(d, 5): +- for each value v_i , i = 1, ..., d
        deltas_tensor = tf.multiply(four, v_expand)

        sketch = tf.tensor_scatter_nd_add(self.zeros_sketch, indices, deltas_tensor)  # shape=(5, 250)

        return sketch

    @staticmethod
    def estimate_euc_norm_squared(sketch):
        """ Estimate the Euclidean norm squared of a vector using its AMS sketch.

        Args:
        - sketch (tf.Tensor): AMS sketch of a vector. Shape=(`depth`, `width`).

        Returns:
        - tf.Tensor: Estimated squared Euclidean norm.
        """

        norm_sq_rows = tf.reduce_sum(tf.square(sketch), axis=1)
        return np.median(norm_sq_rows)


def sketch_rtc(multi_worker_model, w_t0, ams_sketch, epsilon, theta):
    """
    Round Terminating Condition for NaiveFDA

    This function computes the drift of the model's trainable variables from a baseline state, averages the squared
    drift across all replicas, and checks if this average exceeds a specified threshold. The operation ensures a
    unified decision across all replicas, returning a single boolean value that is consistent across the distributed
    context

    Args:
        multi_worker_model (tf.keras.Model): The distributed model being trained.
        w_t0 (tf.Tensor): The last round's model.
        ams_sketch (AmsSketch): An AmsSketch instance (the same vars for all replicas)
        theta (float): The variance threshold.

    Returns:
        bool: A single boolean value, identical across all replicas, indicating whether round should terminate (True)
        or not (False).
    """

    drift = trainable_vars_as_vector(multi_worker_model.trainable_variables) - w_t0

    # ||D(t)_i||^2 , shape = ()
    drift_sq = tf.reduce_sum(tf.square(drift))

    sketch = ams_sketch.sketch_for_vector(drift)

    avg_drift_sq, avg_sketch = tf.distribute.get_replica_context().all_reduce(
        tf.distribute.ReduceOp.MEAN, [drift_sq, sketch]
    )

    return avg_drift_sq - (1. / (1. + epsilon) * AmsSketch.estimate_euc_norm_squared(avg_sketch)) > theta


def sketch_training_loop(strategy, multi_worker_model, multi_worker_dataset,
                         num_epochs, num_steps_per_epoch, theta, per_replica_batch_size, ams_sketch, epsilon):

    epoch_metrics = []

    w_t0 = trainable_vars_as_vector(multi_worker_model.trainable_variables)  # tf.Tensor vector w/ shape=(d,)

    epoch, num_total_rounds, num_total_steps = 0, 0, 0

    while epoch <= num_epochs:
        start_epoch_time = time.time()

        iterator = iter(multi_worker_dataset)
        num_epoch_steps = 0

        while num_epoch_steps <= num_steps_per_epoch:
            strategy.run(fda_step_fn, args=(next(iterator), multi_worker_model, per_replica_batch_size))
            num_epoch_steps += 1
            num_total_steps += 1

            if sketch_rtc(multi_worker_model, w_t0, ams_sketch, epsilon, theta):
                # Synchronization needed - Round terminates
                synced_model_vars = aggregate_models(multi_worker_model.trainable_variables)
                update_distributed_model_vars_from_tensors(multi_worker_model.trainable_variables, synced_model_vars)

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
