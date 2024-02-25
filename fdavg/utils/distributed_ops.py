import tensorflow as tf


# Forced to create it because batch_reduced_to (which is the recommended approch works in TF 2.12 but not it TF 2.7)
def my_reduce(value_to_be_reduced, dest):

    def _merge_fn_mean(distribution, _value_to_be_reduced, _dest):
        return distribution.extended.reduce_to(tf.distribute.ReduceOp.MEAN, _value_to_be_reduced, destinations=_dest)

    return tf.distribute.get_replica_context().merge_call(_merge_fn_mean, args=(value_to_be_reduced, dest))


def aggregate_models(model_variables):
    return [my_reduce(tf.convert_to_tensor(v), v) for v in model_variables]
