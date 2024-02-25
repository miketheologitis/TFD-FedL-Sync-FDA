import tensorflow as tf


def fda_step_fn(inputs, multi_worker_model, per_replica_batch_size):
    """
     Executes a single training step on a batch of data for one replica in a custom distributed training setup,
    applying gradients locally without aggregating them across replicas. This is in contrast to standard distributed
    training practices where gradients are aggregated across all replicas before being applied.

    This function computes the loss and gradients for a batch of data (`inputs`) processed by one replica of
    a distributed model (`multi_worker_model`). Unlike typical distributed training that uses the `global_batch_size`
    for loss averaging across all replicas, here the loss is averaged using `per_replica_batch_size`, reflecting
    the data subset processed by this replica. This adjustment is crucial because gradients are applied locally
    to each replica without being aggregated globally, necessitating subsequent manual synchronization and aggregation
    of model updates across replicas.

    Args:
        inputs (tuple): A tuple containing two elements:
            - x: Input features, a TensorFlow tensor.
            - y: Corresponding labels, a TensorFlow tensor.
        multi_worker_model (tf.keras.Model): The distributed model to be trained. This model should be replicated
            across all devices participating in the training.
        per_replica_batch_size (int): The number of samples processed by each replica in a single step. This value
            is used to correctly scale the loss, ensuring that gradient updates are appropriately scaled for the
            subset of data processed by this replica.

    Notes:
        - `tf.nn.compute_average_loss` is used to average the per-example loss across the `per_replica_batch_size`,
          ensuring that the gradient reflects the correct magnitude for the subset of data processed. This is
          critical because `experimental_aggregate_gradients=False` (or `skip_gradients_aggregation=True` for
          TensorFlow versions 2.11.0 and above) is specified in `optimizer.apply_gradients`, meaning that gradients
          are applied locally without being aggregated across replicas.
        - It's essential to manually synchronize and aggregate model updates across all replicas after local updates
          to maintain model consistency and ensure effective learning.
    """
    x, y = inputs

    with tf.GradientTape() as tape:
        predictions = multi_worker_model(x, training=True)

        per_batch_loss = multi_worker_model.compiled_loss(y, predictions)

        loss = tf.nn.compute_average_loss(per_batch_loss, global_batch_size=per_replica_batch_size)

    grads = tape.gradient(loss, multi_worker_model.trainable_variables)

    multi_worker_model.optimizer.apply_gradients(
        zip(grads, multi_worker_model.trainable_variables),
        experimental_aggregate_gradients=False  # for tf 2.4.2  , skip_gradients_aggregation=True for tf 2.11.0
    )


"""
    return loss


def fda_train_step(strategy, iterator, multi_worker_model, per_replica_batch_size):

    per_replica_losses = strategy.run(fda_step_fn, args=(next(iterator), multi_worker_model, per_replica_batch_size))

    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
"""

def fda_train_step(strategy, iterator, multi_worker_model, per_replica_batch_size):
    return 1

# TODO: Can completely remove returning the loss and directly use ln:57 in the naive training loop as is.
