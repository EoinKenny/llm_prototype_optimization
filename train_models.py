import itertools
import subprocess
import time
from pathlib import Path

prototype_num = 10
num_epochs = 25

def main1():
    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    datasets = ['trec', 'dbpedia', '20newsgroups']
    models   = ['bert', 'electra', 'modern_bert', 'roberta', 'mpnet']
    devices  = ['cuda:0', 'cuda:1', 'cuda:2']           # three GPUs
    seeds    = [0, 1, 2]
    no_head_models = {'llama', 'qwen'}                  # flag applies only here
    out_dir = Path("data")                              # where CSVs are saved

    device_cycle = itertools.cycle(devices)             # round-robin allocator

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
                variants = [
                    dict(baseline=True,  no_llm_head=False),
                    dict(baseline=False, no_llm_head=False)
                ]
                if mdl in no_head_models:
                    variants += [
                        dict(baseline=True,  no_llm_head=True),
                        dict(baseline=False, no_llm_head=True)
                    ]

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
        # pick epochs: 7 for trec, else 3
        # num_epochs = 7 if job['dataset'] == 'trec' else 3
        cmd = (
            f"python src/train_prototype_models.py "
            f"--model='{job['model']}' "
            f"--num_protos={prototype_num} "
            f"--dataset='{job['dataset']}' "
            f"--device='{gpu}' "
            f"--num_epochs={num_epochs} "      # ← uses the variable
            f"--seed={job['seed']}"
        )
        if job['baseline']:
            cmd += " --baseline"
        if job['no_llm_head']:
            cmd += " --no_llm_head"

        tag = []
        tag.append("baseline" if job['baseline'] else "exp")
        if job['no_llm_head']:
            tag.append("no_llm_head")
        tag = "+".join(tag)

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

        time.sleep(5)     # gentle polling interval

    print("✓ All pending training runs completed.")



import itertools
import subprocess
import time
from pathlib import Path

def main2():
    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    datasets = ['trec', 'dbpedia', '20newsgroups']
    models   = ['bert', 'electra', 'modern_bert', 'roberta', 'mpnet']
    devices  = ['cuda:0', 'cuda:1', 'cuda:2']           # three GPUs
    seeds    = [0, 1, 2]
    no_head_models = {'llama', 'qwen'}                  # flag applies only here
    out_dir = Path("data")                              # where CSVs are saved

    device_cycle = itertools.cycle(devices)             # round-robin allocator

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
                variants = [
                    dict(baseline=True,  no_llm_head=False),
                    dict(baseline=False, no_llm_head=False)
                ]
                if mdl in no_head_models:
                    variants += [
                        dict(baseline=True,  no_llm_head=True),
                        dict(baseline=False, no_llm_head=True)
                    ]

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


    # ------------------------------------------------------------------
    # Helper to launch a single process
    # ------------------------------------------------------------------
    def launch(job, gpu):
        # pick epochs: 7 for trec, else 3
        # num_epochs = 7 if job['dataset'] == 'trec' else 3
    
        cmd = (
            f"python src/train_prototype_models.py "
            f"--model='{job['model']}' "
            f"--num_protos={prototype_num} "
            f"--dataset='{job['dataset']}' "
            f"--device='{gpu}' "
            f"--num_epochs={num_epochs} "      # ← uses the variable
            f"--seed={job['seed']}"
        )
        if job['baseline']:
            cmd += " --baseline"
        if job['no_llm_head']:
            cmd += " --no_llm_head"

        tag = []
        tag.append("baseline" if job['baseline'] else "exp")
        if job['no_llm_head']:
            tag.append("no_llm_head")
        tag = "+".join(tag)

        print(f"[{gpu}] Launching ({tag}, seed={job['seed']}): {cmd}")
        return subprocess.Popen(cmd, shell=True)

    # ------------------------------------------------------------------
    # Sequential execution: one job at a time
    # ------------------------------------------------------------------
    for job in jobs:
        gpu = next(device_cycle)
        proc = launch(job, gpu)
        proc.wait()                       # block until the job finishes
        status = "OK" if proc.returncode == 0 else f"ERR ({proc.returncode})"
        print(f"[{gpu}] Job finished ⇒ {status}")
        time.sleep(2)                     # (optional) short breather between runs

    print("✓ All pending training runs completed.")


if __name__ == "__main__":
    main1()
    main2()


    
    
    