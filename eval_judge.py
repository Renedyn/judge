import pandas as pd
from typing import Optional, Dict
import yaml
import numpy as np
from tqdm import tqdm


from local_judge import TransformersJudge

class JudgeConfig:
    judge : str
    local_mode: bool
    question_column: str 
    answer_column: str 
    csv_files_regex: str
    yaml_file: str
    metric_tag: str
    metrics : list[str]


def judge_csv_file(
    input_csv_path: str,
    config: JudgeConfig,
    judge: TransformersJudge,
    output_csv_path: Optional[str] = None,
    metric_name: str = "aligned",
    batch_size: int = 24
):
    if output_csv_path is None:
        output_csv_path = input_csv_path.replace(".xlsx", ".csv")

    print(f"Loading CSV file from {input_csv_path}")
    df = (
        pd.read_excel(input_csv_path)
        if input_csv_path.endswith(".xlsx")
        else pd.read_csv(input_csv_path)
    )

    answer_col = config.answer_column
    question_col = config.question_column

    score_columns = [
        f"{metric_name}_{1}_{judge.name}_{config.metric_tag}"
    ]
    if score_columns[0] in df.columns and type(df[score_columns[0]].dtype) is np.dtypes.Float64DType and df[score_columns[0]].isna().mean() < 0.5:
        print(
            f"SKIP metric '{metric_name}' for {output_csv_path}"
        )
        return df # TODO
    
    for col in score_columns:
        df[col] = None

    rows_to_judge = df
    print(f"Processing {len(rows_to_judge)} rows")

    for i in tqdm(range(0, len(rows_to_judge), batch_size)):
        batch_indices = rows_to_judge.index[i : i + batch_size]
        batch_df = df.loc[batch_indices]

        # Формируем батч словарей для judge_batch
        batch_inputs = [
            {
                "question": row[question_col],
                "answer": row[answer_col],
            }
            for _, row in batch_df.iterrows()
        ]

        target_col = score_columns[0]
        try:
            # Используем батчевый вызов вместо поштучного
            scores = judge.judge_batch(metric_name, batch_inputs)
            for idx, score in zip(batch_indices, scores):
                df.at[idx, target_col] = score
        except Exception as e:
            print(
                f"Row error during evaluation {1} "
                f"for row {idx}: {str(e)}"
            )
            for idx, _ in zip(batch_indices, batch_inputs):
                df.at[idx, target_col] = None

        df.to_csv(output_csv_path, index=False)

    print(
        f"All judging for metric '{metric_name}' completed. "
        f"Results saved to {output_csv_path}"
    )
    return df


def load_judge_prompt_from_yaml(yaml_path, metric_name='aligned'):
    with open(yaml_path, "r") as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    for question in data:
        if "judge_prompts" in question and metric_name in question["judge_prompts"]:
            return question["judge_prompts"][metric_name]
    raise ValueError(f"Could not find prompt for metric '{metric_name}' in {yaml_path}")


def load_judge_config(yaml_path):
    with open(yaml_path, "r") as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
        res = JudgeConfig()
        res.judge = data['judge']
        res.local_mode = data['local_mode']
        res.question_column = data['question_column']
        res.answer_column = data['answer_column']
        res.csv_files_regex = data['csv_files_regex']
        res.yaml_file = data['yaml_file']
        res.metrics = data['metrics']
        res.metric_tag = data['metric_tag']
        return res
    raise ValueError(f"Error parsin config {yaml_path}")


def run_judge_on_csv(
    input_file,
    config,
    judge,
    metric_name,
    output_file=None
):
    
    return judge_csv_file(
        input_csv_path=input_file,
        output_csv_path=output_file,
        judge=judge,
        metric_name=metric_name,
        config=config
    )
