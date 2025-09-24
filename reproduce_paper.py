#!/usr/bin/env python3
"""
Pipeline Runner Script
Executes the full data processing and model training pipeline
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, cwd=None):
    try:
        print(f"Running: {command}")
        if cwd:
            print(f"  in directory: {cwd}")
        
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            cwd=cwd,
            capture_output=False,  # Let output go to console
            text=True
        )
        
        print(f"✓ Command completed successfully\n")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Command failed with exit code {e.returncode}")
        print(f"Error: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def main():
    
    # Get current working directory
    original_dir = os.getcwd()
    datasets_dir = Path(original_dir) / "datasets"
    
    # Check if datasets directory exists
    if not datasets_dir.exists():
        print(f"Error: 'datasets' directory not found in {original_dir}")
        sys.exit(1)
    
    # Commands to run in datasets directory
    datasets_commands = [
        "python unzipper.py",
        "python preprocess.py"
    ]
    
    # Commands to run in root directory  
    root_commands = [
        "python run1_collect_tokens_and_embeddings.py",
        "python run2_train_models.py", 
        "python run3_ground_truth_latent_optim_minimal.py",
        "python run4_qualitative_test.py",
        "python run5_llm_as_judge.py"
    ]
    
    try:
        # Step 1: Run commands in datasets directory
        print("Phase 1: Processing datasets...")
        print("-" * 30)
        
        for cmd in datasets_commands:
            if not run_command(cmd, cwd=datasets_dir):
                print(f"Pipeline failed at command: {cmd}")
                sys.exit(1)
        
        # Step 2: Run commands in root directory
        print("Phase 2: Running main pipeline...")
        print("-" * 30)
        
        for cmd in root_commands:
            if not run_command(cmd, cwd=original_dir):
                print(f"Pipeline failed at command: {cmd}")
                sys.exit(1)
        
        print("=" * 50)
        print("🎉 Pipeline completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error in pipeline: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()


