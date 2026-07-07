"""Sequential entry point for the train stage."""
from src.toy_pipeline import activate_from_argv, run_stage

if __name__ == "__main__":
    if activate_from_argv():
        run_stage("train")
    else:
        from src.train_all import main
        main()
