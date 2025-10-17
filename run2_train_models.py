import itertools
import subprocess
import time
from pathlib import Path
from config import DATASETS, MODELS, SEEDS, devices

def main():
    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    datasets = DATASETS
    models   = MODELS   
    seeds    = SEEDS
    out_dir = Path("data")                          # where CSVs are saved
    device_cycle = itertools.cycle(devices)         # round-robin allocator
    
    # ------------------------------------------------------------------
    # Build the job list
    # ------------------------------------------------------------------
    jobs = []
    for ds in datasets:
        for mdl in models:
            for sd in seeds:
                jobs.append(dict(dataset=ds, model=mdl, seed=sd))
    
    print(f"⇒ {len(jobs)} total jobs to launch.")
    running = []
    
    # ------------------------------------------------------------------
    # Helper to launch a single process
    # ------------------------------------------------------------------
    def launch(job, gpu):
        
        if job['dataset'] in ['imdb']:
            num_epochs = 5
            prototype_num = 3
        elif job['dataset'] in ['amazon_reviews']:
            num_epochs = 5
            prototype_num = 3
        elif job['dataset'] in ['agnews']:
            num_epochs = 10
            prototype_num = 3
        elif job['dataset'] in ['20newsgroups']:
            num_epochs = 20
            prototype_num = 1
        elif job['dataset'] in ['dbpedia']:
            num_epochs = 15
            prototype_num = 1
        elif job['dataset'] in ['trec']:
            num_epochs = 100
            prototype_num = 1
        else:
            raise NameError(f"Wrong dataset: {job['dataset']}")
            
        print(f"Doing {num_epochs} epochs for {job['dataset']}")
            
        cmd = (
            f"python src/train_prototype_models.py "
            f"--model={job['model']} "
            f"--num_protos={prototype_num} "
            f"--dataset={job['dataset']} "
            f"--device={gpu} "
            f"--num_epochs={num_epochs} "
            f"--seed={job['seed']} "
            f"--use_class_weights"
        )
        
        print(f"[{gpu}] Launching (model={job['model']}, dataset={job['dataset']}, seed={job['seed']}): {cmd}")
        return subprocess.Popen(cmd, shell=True)
    
    # ------------------------------------------------------------------
    # Scheduler loop
    # ------------------------------------------------------------------
    while jobs or running:
        # Fill idle GPUs
        while jobs and len(running) < len(devices):
            job = jobs.pop(0)
            gpu = next(device_cycle)
            running.append((launch(job, gpu), gpu))
        
        # Reap finished jobs
        for proc, gpu in running[:]:
            if proc.poll() is not None:
                running.remove((proc, gpu))
                status = "OK" if proc.returncode == 0 else f"ERR ({proc.returncode})"
                print(f"[{gpu}] Job finished ⇒ {status}")
        
        time.sleep(5)
    
    print("✓ All training runs completed.")

if __name__ == "__main__":
    main()
