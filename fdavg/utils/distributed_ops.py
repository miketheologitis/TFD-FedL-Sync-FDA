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

    synced_model_vars = average_model_trainable_variables(multi_worker_model)

    update_model_vars(multi_worker_model.trainable_variables, synced_model_vars)
