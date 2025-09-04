#!/usr/bin/env python3
"""
Run the entire healthcare anonymization project from one place.

Features:
- Run the healthcare anonymization pipeline
- Optionally train the CGAN with specified epochs/batch size
- Optionally generate sample anonymized data using the latest trained CGAN

Usage examples:
  python run_project.py --input data/unified_5000_records.csv --records 1000
  python run_project.py --input data/unified_5000_records.csv --records 1000 --train-cgan --epochs 500 --batch-size 32 --model-dir models
  python run_project.py --input data/unified_5000_records.csv --records 1000 --generate-samples
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd: list[str]) -> None:
    print("\n$", " ".join(cmd))
    completed = subprocess.run(cmd, shell=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def run_pipeline(input_path: str, records: int | None) -> None:
    cmd = [sys.executable, "healthcare_anonymization_pipeline.py", "--input", input_path]
    if records:
        cmd += ["--records", str(records)]
    run_cmd(cmd)


def train_cgan(input_path: str, epochs: int, batch_size: int, records: int | None, model_dir: str) -> None:
    # Reuse the improved training script
    cmd = [
        sys.executable,
        "train_cgan_improved.py",
        "--input", input_path,
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--model-dir", model_dir,
    ]
    if records:
        cmd += ["--records", str(records)]
    run_cmd(cmd)


def generate_samples() -> None:
    # Use the dedicated generator script
    cmd = [sys.executable, "generate_sample_data.py"]
    run_cmd(cmd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run healthcare anonymization pipeline and optional CGAN training")
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument("--records", type=int, default=None, help="Limit the number of records to process")

    # Pipeline toggle
    parser.add_argument("--no-pipeline", action="store_true", help="Skip running the anonymization pipeline")

    # CGAN training options
    parser.add_argument("--train-cgan", action="store_true", help="Train CGAN after pipeline")
    parser.add_argument("--epochs", type=int, default=500, help="CGAN training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="CGAN training batch size")
    parser.add_argument("--model-dir", type=str, default="models", help="Directory to save CGAN models")

    # Sample generation
    parser.add_argument("--generate-samples", action="store_true", help="Generate sample synthetic data using latest CGAN")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 70)
    print("HEALTHCARE ANONYMIZATION - PROJECT RUNNER")
    print("=" * 70)
    print(f"Input file          : {args.input}")
    print(f"Records             : {args.records or 'ALL'}")
    print(f"Run pipeline        : {not args.no_pipeline}")
    print(f"Train CGAN          : {args.train_cgan}")
    print(f"CGAN epochs         : {args.epochs}")
    print(f"CGAN batch size     : {args.batch_size}")
    print(f"Model dir           : {args.model_dir}")
    print(f"Generate samples    : {args.generate_samples}")
    print("=" * 70)

    # Ensure input exists
    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    # 1) Run pipeline (unless skipped)
    if not args.no_pipeline:
        print("\n--- Running Healthcare Anonymization Pipeline ---")
        run_pipeline(str(input_path), args.records)

    # 2) Train CGAN (optional)
    if args.train_cgan:
        print("\n--- Training CGAN ---")
        train_cgan(str(input_path), args.epochs, args.batch_size, args.records, args.model_dir)

    # 3) Generate sample data (optional)
    if args.generate_samples:
        print("\n--- Generating Sample Synthetic Data ---")
        generate_samples()

    print("\nAll requested steps completed successfully.")


if __name__ == "__main__":
    main()


