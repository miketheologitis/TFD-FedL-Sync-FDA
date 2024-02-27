import argparse
import os
import pandas as pd
from fdavg.utils.manage_combinations import get_experiment_information, derive_extra_params, get_test_id
from fdavg.strategies.multi_worker_mirrored_training import multi_worker_mirrored_train
from fdavg.utils.cluster_configuration import configure_cluster
from fdavg.metrics.metrics import process_epoch_metrics_with_test_id, process_step_metrics_with_test_id

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
TMP_DIR = '../metrics/tmp'
EPOCH_METRICS_PATH = os.path.normpath(os.path.join(SCRIPT_DIR, f'{TMP_DIR}/epoch_metrics'))
STEP_METRICS_PATH = os.path.normpath(os.path.join(SCRIPT_DIR, f'{TMP_DIR}/step_metrics'))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--comb_file_id', type=int, help="The combinations prefix, i.e., <PREFIX>.json")
    parser.add_argument('--exper_id', type=int, help="The Exper. ID (0, 1, 2, ...)")
    args = parser.parse_args()

    exper_info = get_experiment_information(args.comb_file_id, args.exper_id)

    configure_cluster(exper_info)
    derive_extra_params(exper_info)
    test_id = get_test_id(exper_info)

    if exper_info['task_index'] == 0:
        print(f"{test_id}\n")

    # Run experiment
    epoch_metrics = multi_worker_mirrored_train(exper_info)

    epoch_metrics_df = pd.DataFrame(process_epoch_metrics_with_test_id(epoch_metrics, test_id))
    epoch_metrics_df.to_csv(
        f"{EPOCH_METRICS_PATH}/task{exper_info['task_index']}_{exper_info['exper_filename']}.csv",
        index=False
    )

    if exper_info['task_index'] == 0:
        print("\nFinished Experiment!\n")

