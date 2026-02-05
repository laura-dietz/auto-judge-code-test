#!/usr/bin/env python3
"""
Debug wrapper for running judges with limits and detailed logging.

Usage:
    python run_judge_debug.py \
        --judge {non_llm|umbrela} \
        --rag-topics topics.jsonl \
        --rag-responses runs/ \
        --out-dir output/ \
        --max-topics 5 \
        --max-runs 3 \
        --debug-log debug.log
"""
import argparse
import sys
from pathlib import Path
import json
from datetime import datetime
from collections import defaultdict
import os
import yaml


def load_topics(topics_file, max_topics=None):
    """Load topics from jsonl file."""
    topics = []
    with open(topics_file) as f:
        for i, line in enumerate(f):
            if max_topics and i >= max_topics:
                break
            topics.append(json.loads(line))
    return topics


def load_runs(runs_dir, max_runs=None):
    """Load run files from directory."""
    runs_dir = Path(runs_dir)
    run_files = sorted(runs_dir.glob('*'))
    if max_runs:
        run_files = run_files[:max_runs]
    return run_files


def filter_responses_by_topic(responses_file, topic_ids):
    """Filter responses to only include specified topics."""
    filtered = []
    with open(responses_file) as f:
        for line in f:
            data = json.loads(line)
            if data.get('metadata', {}).get('topic_id') in topic_ids:
                filtered.append(data)
    return filtered


def create_filtered_dataset(runs_dir, topics, max_runs, output_dir):
    """Create filtered dataset with limited topics and runs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    topic_ids = {t.get('topic_id') or t.get('request_id') or t.get('id') for t in topics}

    # Create filtered topics file
    topics_file = output_dir / "topics_filtered.jsonl"
    with open(topics_file, 'w') as f:
        for topic in topics:
            f.write(json.dumps(topic) + '\n')

    # Create filtered runs directory
    runs_out = output_dir / "runs_filtered"
    runs_out.mkdir(exist_ok=True)

    run_files = load_runs(runs_dir, max_runs)
    for run_file in run_files:
        responses = filter_responses_by_topic(run_file, topic_ids)
        out_file = runs_out / run_file.name
        with open(out_file, 'w') as f:
            for resp in responses:
                f.write(json.dumps(resp) + '\n')

    return topics_file, runs_out


def setup_logging(log_file):
    """Setup debug log file."""
    with open(log_file, 'w') as f:
        f.write("=== Judge Debug Run ===\n")
        f.write(f"Started: {datetime.now()}\n")
        f.write("=" * 80 + "\n\n")


def log_config(log_file, config):
    """Log configuration."""
    with open(log_file, 'a') as f:
        f.write("\n=== Configuration ===\n")
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")


def create_llm_config_from_env(output_file):
    """Create LLM config file from environment variables.

    Reads OPENAI_BASE_URL, OPENAI_MODEL, and OPENAI_API_KEY from environment.
    """
    base_url = os.getenv('OPENAI_BASE_URL')
    model = os.getenv('OPENAI_MODEL')
    api_key = os.getenv('OPENAI_API_KEY')

    if not base_url or not model:
        raise ValueError(
            "OPENAI_BASE_URL and OPENAI_MODEL environment variables must be set. "
            "Example:\n"
            "  export OPENAI_BASE_URL='https://api.together.xyz/v1'\n"
            "  export OPENAI_MODEL='meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo'\n"
            "  export OPENAI_API_KEY='your-key'"
        )

    config = {
        'base_url': base_url,
        'model': model,
    }

    if api_key:
        config['api_key'] = api_key

    with open(output_file, 'w') as f:
        yaml.dump(config, f)

    return output_file


def main():
    parser = argparse.ArgumentParser(description='Run judge with limits and debug logging')
    parser.add_argument('--judge', required=True, choices=['non_llm', 'umbrela'],
                       help='Which judge to run')
    parser.add_argument('--rag-topics', required=True,
                       help='Path to topics file')
    parser.add_argument('--rag-responses', required=True,
                       help='Path to runs directory')
    parser.add_argument('--out-dir', required=True,
                       help='Output directory')
    parser.add_argument('--workflow', default='workflow.yml',
                       help='Workflow config file')
    parser.add_argument('--llm-config', help='LLM config file (for umbrela)')
    parser.add_argument('--use-env-llm', action='store_true',
                       help='Create LLM config from environment variables (OPENAI_BASE_URL, OPENAI_MODEL, OPENAI_API_KEY)')
    parser.add_argument('--max-topics', type=int,
                       help='Maximum number of topics to process')
    parser.add_argument('--max-runs', type=int,
                       help='Maximum number of runs to process')
    parser.add_argument('--debug-log',
                       help='Debug log file path')

    args = parser.parse_args()

    # Setup logging if requested
    if args.debug_log:
        setup_logging(args.debug_log)
        log_config(args.debug_log, vars(args))

    # Load and filter data
    print(f"Loading topics from {args.rag_topics}...")
    topics = load_topics(args.rag_topics, args.max_topics)
    print(f"  Loaded {len(topics)} topics")

    # Create filtered dataset
    print(f"Creating filtered dataset...")
    temp_dir = Path(args.out_dir) / "temp_filtered"
    topics_file, runs_dir = create_filtered_dataset(
        args.rag_responses, topics, args.max_runs, temp_dir
    )
    print(f"  Topics: {topics_file}")
    print(f"  Runs: {runs_dir}")

    # Build command for actual judge
    # Convert relative paths to absolute
    workflow_path = Path(args.workflow).resolve()
    out_dir_path = Path(args.out_dir).resolve()

    judge_dir = Path(__file__).parent / args.judge.replace('_', '_')
    if args.judge == 'non_llm':
        judge_dir = Path(__file__).parent / "non_llm_judge"
        judge_script = judge_dir / "non_llm_judge.py"
        cmd = [
            sys.executable, str(judge_script),
            '--workflow', str(workflow_path),
            '--rag-topics', str(topics_file.resolve()),
            '--rag-responses', str(runs_dir.resolve()),
            '--out-dir', str(out_dir_path),
        ]
    else:  # umbrela
        judge_dir = Path(__file__).parent / "umbrela"
        judge_script = judge_dir / "umbrela_judge.py"

        # Determine LLM config path
        if args.use_env_llm:
            # Create LLM config from environment variables
            llm_config_path = temp_dir / 'llm-config-env.yml'
            print(f"Creating LLM config from environment variables...")
            create_llm_config_from_env(llm_config_path)
            print(f"  Config: {llm_config_path}")
        elif args.llm_config:
            llm_config_path = Path(args.llm_config).resolve()
        else:
            llm_config_path = judge_dir / 'llm-config.yml'

        cmd = [
            sys.executable, str(judge_script), 'run',
            '--rag-responses', str(runs_dir.resolve()),
            '--rag-topics', str(topics_file.resolve()),
            '--workflow', str(workflow_path),
            '--llm-config', str(llm_config_path),
            '--out-dir', str(out_dir_path),
        ]

    print(f"\nRunning judge...")
    print(f"Command: {' '.join(cmd)}\n")

    import subprocess
    result = subprocess.run(cmd, cwd=judge_dir)

    if result.returncode == 0:
        print(f"\n✓ Judge completed successfully")
        print(f"  Output: {args.out_dir}")
        if args.debug_log:
            print(f"  Debug log: {args.debug_log}")
    else:
        print(f"\n✗ Judge failed with exit code {result.returncode}")
        sys.exit(result.returncode)


if __name__ == '__main__':
    main()
