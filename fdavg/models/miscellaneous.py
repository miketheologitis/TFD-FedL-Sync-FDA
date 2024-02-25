import tensorflow as tf


def trainable_vars_as_vector(trainable_variables):
    """This is a helper function used to trasform a list of vectors/tensors into a column vector/tensor"""
    return tf.concat([tf.reshape(var, [-1]) for var in trainable_variables], axis=0)


def update_distributed_model_vars_from_tensors(distr_model_vars, new_values):
    """
    Updates TensorFlow MirroredVariables with new values from tensors.

    This function is designed for use in distributed training environments
    utilizing TensorFlow's distribution strategies (e.g., MirroredStrategy).
    It ensures that each replica's variable is updated with the same new value,
    maintaining consistency across all devices.

    Args:
        distr_model_vars (List[MirroredVariable]): A list of MirroredVariable
            instances representing the model's trainable variables. Each
            MirroredVariable corresponds to a model variable distributed
            across all replicas/devices.
        new_values (List[tf.Tensor]): A list of tensors containing the new
            values for each trainable variable. The order of tensors must
            match the order of variables in `model_train_vars`. Each tensor's
            shape and dtype must be compatible with the corresponding variable.
    """
    for old_var, new_value in zip(distr_model_vars, new_values):
        for v in old_var.values:
            v.assign(new_value)


def update_model_vars(old_model_vars, new_model_vars):
    for old_var, new_var in zip(old_model_vars, new_model_vars):
        old_var.assign(new_var)


