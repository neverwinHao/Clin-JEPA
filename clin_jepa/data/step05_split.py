"""Step 05: split patients 70/15/15 into train/val/test, stratified by mortality and Sepsis-3."""

import argparse
import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from clin_jepa.data.helpers.io_utils import resolve_table_path
from clin_jepa.utils import ensure_dir, load_config, set_seed, setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths_config", default="configs/data/paths.yaml")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    setup_logging()
    set_seed(args.seed)
    paths_config = load_config(args.paths_config)
    cohort = pd.read_csv(resolve_table_path(paths_config, "output.cohort") / "cohort.csv")

    subjects = (
        cohort.groupby("subject_id")
        .agg(mortality=("hospital_expire_flag", "max"), sepsis3=("sepsis3", "max"))
        .reset_index()
    )
    subjects["strat_label"] = (
        subjects["mortality"].astype(int).astype(str)
        + "_"
        + subjects["sepsis3"].astype(int).astype(str)
    )

    train_subj, temp_subj, _, temp_strat = train_test_split(
        subjects["subject_id"].values, subjects["strat_label"].values,
        test_size=0.30, random_state=args.seed, stratify=subjects["strat_label"].values,
    )
    val_subj, test_subj = train_test_split(
        temp_subj, test_size=0.50, random_state=args.seed, stratify=temp_strat,
    )
    train_set, val_set = set(train_subj), set(val_subj)

    splits = cohort[["stay_id", "subject_id"]].copy()
    splits["split"] = splits["subject_id"].map(
        lambda s: "train" if s in train_set else "val" if s in val_set else "test"
    )

    output_path = ensure_dir(resolve_table_path(paths_config, "output.splits")) / "splits.csv"
    splits.to_csv(output_path, index=False)
    logger.info("wrote %s: %s", output_path, splits["split"].value_counts().to_dict())


if __name__ == "__main__":
    main()
