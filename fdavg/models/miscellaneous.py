import tensorflow as tf


def trainable_vars_as_vector(trainable_variables):
    """This is a helper function used to trasform a list of vectors/tensors into a column vector/tensor"""
    return tf.concat([tf.reshape(var, [-1]) for var in trainable_variables], axis=0)


def update_model_vars(old_model_vars, new_model_vars):
    for old_var, new_var in zip(old_model_vars, new_model_vars):
        old_var.assign(new_var)


