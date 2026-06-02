import os
import glob 

import sys
import tqdm
#sys.path.append('/content')

import argparse
from gen_eval_util import judge_responses
from local_judge import TransformersJudge
from judge_azure import OpenAiJudge
from eval_judge import load_judge_config, load_judge_prompt_from_yaml, JudgeConfig

def process_single_file(csv_file, judge, config : JudgeConfig):
    """Process a single CSV file."""
    if not os.path.exists(csv_file):
        print(f"File {csv_file} not found!")
        return None
    print(f"Starting evaluation for file: {csv_file}")
    try:
        df = judge_responses(
            save_path=csv_file,
            judge=judge,
            config=config
        )
        print(f"Evaluation for file {csv_file} completed! Results saved to CSV.")
        return df
    except Exception as e:
        print(f"Error processing file {csv_file}: {str(e)}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="yaml config file") 
    args = parser.parse_args()
    config = load_judge_config(args.config)
    

    # Configuration
    #csv_files = glob.glob("content/*grok*") + glob.glob("content/*kimi*") # 
    csv_files = glob.glob(config.csv_files_regex)  # find all .csv files in the current directory
    yaml_file = config.yaml_file
    metrics = config.metrics

    if not os.path.exists(yaml_file):
        print(f"File {yaml_file} not found!")
        return

    if not csv_files:
        print("No CSV files found!")
        return

    print(f"Found {len(csv_files)} CSV files. Starting processing...")

    judge_name = config.judge
    is_local = config.local_mode
    prompt_templates = {metric : load_judge_prompt_from_yaml(yaml_file, metric) for metric in metrics}
    if is_local:
        judge = TransformersJudge(model=judge_name, name=judge_name, prompt_templates=prompt_templates)
    else:
        judge = OpenAiJudge(deployment=judge_name, prompt_templates=prompt_templates)

    results = [
        process_single_file(csv_file, judge, config)
        for csv_file in csv_files
    ]

    successful = sum(1 for r in results if r is not None and not isinstance(r, Exception))
    print(f"\nProcessing complete! Successfully processed: {successful}/{len(csv_files)} files.")

if __name__ == "__main__":
    main()
