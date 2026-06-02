import pandas as pd
import yaml

from eval_judge import run_judge_on_csv


def judge_responses(
    save_path,
    judge,
    config
):
    for metric in config.metrics:
        df = run_judge_on_csv(
            input_file=save_path,
            judge=judge,
            config=config,
            metric_name=metric,
        )
    return df
