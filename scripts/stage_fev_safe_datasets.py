from __future__ import annotations

import argparse
import logging

from datasets import load_dataset

from toto_interp.fev_tasks import list_fev_tasks
from toto_interp.transfer import FEV_DATASET_REPO_ID


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage all paper-safe FEV dataset configurations into the HuggingFace cache."
    )
    parser.add_argument("--max-configs", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    configs = sorted({task.dataset_config for task in list_fev_tasks(safe_only=True)})
    if args.max_configs > 0:
        configs = configs[: args.max_configs]
    for index, config in enumerate(configs, start=1):
        dataset = load_dataset(FEV_DATASET_REPO_ID, config, split="train")
        logging.info("[%d/%d] staged %s (%d rows)", index, len(configs), config, len(dataset))


if __name__ == "__main__":
    main()
