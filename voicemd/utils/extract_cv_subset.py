import argparse
import os
import shutil
import pandas as pd


def copy_subset(cv_path, subset_path):
    metadata = pd.read_csv(
        os.path.join(cv_path,
                     'cv_train_metadata.csv')
    )

    for row, sample in metadata.iterrows():
        src = os.path.join(cv_path, 'clips', sample['path'])
        dest = os.path.join(subset_path, 'clips')
        shutil.copy(src, dest)

    metadata.to_csv(os.path.join(subset_path, 'cv_metadata_balanced.csv'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--commonvoice_path", help="path to commonvoice dataset", required=True
    )
    parser.add_argument(
        "--sample_dir", help="path to folder to store samples", required=True
    )
    args = parser.parse_args()

    copy_subset(args.commonvoice_path, args.sample_dir)
