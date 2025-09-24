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
    # Build the job list (skip any that already have a CSV)
    # ------------------------------------------------------------------
    jobs = []
    def result_file(mdl, ds, sd, baseline, no_head):
        """Return Path object for the expected CSV output."""
        return out_dir / f"{mdl}_{ds}_protos{prototype_num}_baseline{baseline}_seed{sd}_no_llm_head{no_head}.csv"

    for ds in datasets:
        for mdl in models:
            for sd in seeds:
                # Only keep baseline=True
                variants = [dict(baseline=True, no_llm_head=False)]

                for var in variants:
                    csv_path = result_file(mdl, ds, sd, var['baseline'], var['no_llm_head'])
                    if csv_path.exists():
                        print(f"✓ Skipping completed run  → {csv_path.name}")
                        continue

                    jobs.append(dict(dataset=ds, model=mdl, seed=sd,
                                     baseline=var['baseline'],
                                     no_llm_head=var['no_llm_head']))

    print(f"⇒ {len(jobs)} jobs remain to be launched.")
    if not jobs:
        print("All variants already finished – nothing to do!")
        return

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
            raise NameError('Wrong dataset')
            
        print('doing', num_epochs, 'epochs for', job['dataset'])
            
        cmd = (
            f"python src/train_prototype_models.py "
            f"--model='{job['model']}' "
            f"--num_protos={prototype_num} "
            f"--dataset='{job['dataset']}' "
            f"--device='{gpu}' "
            f"--num_epochs={num_epochs} "
            f"--seed={job['seed']} "
            f"--baseline"
        )
        if job['no_llm_head']:
            cmd += " --no_llm_head"

        tag = "baseline" + ("+no_llm_head" if job['no_llm_head'] else "")
        print(f"[{gpu}] Launching ({tag}, seed={job['seed']}): {cmd}")
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

    print("✓ All pending training runs completed.")


if __name__ == "__main__":
    main()

