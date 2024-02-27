import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.densenet import preprocess_input
import os

MNIST_N_TRAIN = 60_000
CIFAR10_N_TRAIN = 50_000

script_dir = os.path.dirname(os.path.realpath(__file__))

mnist_dir = 'data/mnist'
mnist_data = os.path.normpath(os.path.join(script_dir, f'{mnist_dir}/mnist.npz'))

cifar10_dir = 'data/cifar10'
cifar10_part1_data = os.path.normpath(os.path.join(script_dir, f'{cifar10_dir}/cifar10_part1.npz'))
cifar10_part2_data = os.path.normpath(os.path.join(script_dir, f'{cifar10_dir}/cifar10_part2.npz'))


def load_mnist_from_local_npz():
    with np.load(mnist_data) as data:
        return (data['X_train'], data['y_train']), (data['X_test'], data['y_test'])


def mnist_load_data():
    (X_train, y_train), (X_test, y_test) = load_mnist_from_local_npz()

    X_train, X_test = X_train / 255.0, X_test / 255.0

    return X_train, y_train, X_test, y_test


def mnist_worker_dataset(num_workers, i):
    X_train, y_train, _, _ = mnist_load_data()

    X_train_lst = np.array_split(X_train, num_workers)
    y_train_lst = np.array_split(y_train, num_workers)

    return tf.data.Dataset.from_tensor_slices((X_train_lst[i], y_train_lst[i]))


def mnist_worker_test_dataset(num_workers, i):
    _, _, X_test, y_test = mnist_load_data()

    X_test_lst = np.array_split(X_test, num_workers)
    y_test_lst = np.array_split(y_test, num_workers)

    return tf.data.Dataset.from_tensor_slices((X_test_lst[i], y_test_lst[i]))


def mnist_dataset_fn(global_batch_size, input_context):
    batch_size = input_context.get_per_replica_batch_size(global_batch_size)

    dataset = mnist_worker_dataset(input_context.num_input_pipelines, input_context.input_pipeline_id)

    shuffle_size = dataset.cardinality()

    return dataset.shuffle(shuffle_size).batch(batch_size).prefetch(10)


def mnist_test_dataset_fn(global_batch_size, input_context):
    batch_size = input_context.get_per_replica_batch_size(global_batch_size)

    dataset = mnist_worker_test_dataset(input_context.num_input_pipelines, input_context.input_pipeline_id)

    return dataset.batch(batch_size).prefetch(1)


# CIFAR-10

def load_cifar10_from_local_npz():
    with np.load(cifar10_part1_data) as data1:
        with np.load(cifar10_part2_data) as data2:
            X_train = np.concatenate((data1['X_train'], data2['X_train']))
            y_train = np.concatenate((data1['y_train'], data2['y_train']))
            X_test = np.concatenate((data1['X_test'], data2['X_test']))
            y_test = np.concatenate((data1['y_test'], data2['y_test']))

            return (X_train, y_train), (X_test, y_test)


def cifar10_load_data():
    (X_train, y_train), (X_test, y_test) = load_cifar10_from_local_npz()

    X_train = preprocess_input(X_train)
    X_test = preprocess_input(X_test)

    return X_train, y_train, X_test, y_test


def cifar10_worker_dataset(num_workers, i):
    X_train, y_train, _, _ = cifar10_load_data()

    X_train_lst = np.array_split(X_train, num_workers)
    y_train_lst = np.array_split(y_train, num_workers)

    return tf.data.Dataset.from_tensor_slices((X_train_lst[i], y_train_lst[i]))


def cifar10_worker_test_dataset(num_workers, i):
    _, _, X_test, y_test = cifar10_load_data()

    X_test_lst = np.array_split(X_test, num_workers)
    y_test_lst = np.array_split(y_test, num_workers)

    return tf.data.Dataset.from_tensor_slices((X_test_lst[i], y_test_lst[i]))


def cifar10_dataset_fn(global_batch_size, input_context):
    batch_size = input_context.get_per_replica_batch_size(global_batch_size)

    dataset = cifar10_worker_dataset(input_context.num_input_pipelines, input_context.input_pipeline_id)

    shuffle_size = dataset.cardinality()

    return dataset.shuffle(shuffle_size).batch(batch_size).prefetch(10)


def cifar10_test_dataset_fn(global_batch_size, input_context):
    batch_size = input_context.get_per_replica_batch_size(global_batch_size)

    dataset = cifar10_worker_test_dataset(input_context.num_input_pipelines, input_context.input_pipeline_id)

    return dataset.batch(batch_size).prefetch(1)
