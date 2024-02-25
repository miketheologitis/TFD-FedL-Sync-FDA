from collections import namedtuple

"""
TestId namedtuple:
Attributes:
    - dataset_name (str): Name of the dataset used.
    - strat_name (str): Strat name.
    - num_clients (int): Number of clients in the federated network.
    - batch_size (int): Batch size for training.
    - theta (float): The threshold value for FDA.
    - nn_name (str): The NN name.
"""
# Define a named tuple to represent test IDs for different experiments.
TestId = namedtuple(
        'TestId',
        [
            "dataset_name", "strat_name", "num_clients", "batch_size", "theta", "nn_name"
         ]
)

"""
EpochMetrics namedtuple:
Attributes:
    - epoch (int): The epoch number.
    - total_rounds (int): The total number of rounds at the end of this epoch.
    - total_fda_steps (int): Total FDA steps taken till this epoch.
    - accuracy (float): Model's accuracy at the end of this epoch.
"""
# Define a named tuple to store epoch-specific metrics in federated learning.
EpochMetrics = namedtuple(
    "EpochMetrics", ["epoch", "total_rounds", "total_fda_steps", "accuracy", "train_loss", "train_accuracy"]
)

# Extend the EpochMetrics and RoundMetrics namedtuples to include TestId.
EpochMetricsWithId = namedtuple('EpochMetricsWithId', TestId._fields + EpochMetrics._fields)

"""
EpochMetrics namedtuple:
Attributes:
    - epoch (int): The epoch number.
    - total_rounds (int): The total number of rounds at the end of this epoch.
    - total_fda_steps (int): Total FDA steps taken till this epoch.
    - accuracy (float): Model's accuracy at the end of this epoch.
"""
# Define a named tuple to store epoch-specific metrics in federated learning.
StepMetrics = namedtuple(
    "EpochMetrics", ["epoch", "step", "time_ms"]
)

# Extend the EpochMetrics and RoundMetrics namedtuples to include TestId.
StepMetricsWithId = namedtuple('StepMetricsWithId', TestId._fields + StepMetrics._fields)


def process_epoch_metrics_with_test_id(epoch_metrics_list, test_id):
    """
    Process the given epoch and round metrics lists to append TestId.

    Args:
    - epoch_metrics_list (list of EpochMetrics): List of epoch metrics.
    - test_id (TestId): An instance of TestId namedtuple.

    Returns:
    - tuple: (epoch_metrics_with_test_id, round_metrics_with_test_id)
    """
    epoch_metrics_with_test_id = [
        EpochMetricsWithId(*test_id, *epoch_metrics)
        for epoch_metrics in epoch_metrics_list
    ]

    return epoch_metrics_with_test_id


def process_step_metrics_with_test_id(step_metrics_list, test_id):
    """
    Process the given epoch and round metrics lists to append TestId.

    Args:
    - epoch_metrics_list (list of EpochMetrics): List of epoch metrics.
    - test_id (TestId): An instance of TestId namedtuple.

    Returns:
    - tuple: (epoch_metrics_with_test_id, round_metrics_with_test_id)
    """
    step_metrics_with_test_id = [
        StepMetricsWithId(*test_id, *epoch_metrics)
        for epoch_metrics in step_metrics_list
    ]

    return step_metrics_with_test_id
