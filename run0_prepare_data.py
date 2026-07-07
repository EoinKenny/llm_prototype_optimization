"""Sequential entry point for the prepare data stage."""
from src.toy_pipeline import activate_from_argv, run_stage

if __name__ == "__main__":
    if activate_from_argv():
        run_stage("prepare_data")
    else:
        from src.prepare_data import main
        main()
