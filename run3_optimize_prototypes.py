"""Sequential entry point for the optimize stage."""
from src.toy_pipeline import activate_from_argv, run_stage

if __name__ == "__main__":
    if activate_from_argv():
        run_stage("optimize")
    else:
        from src.optimize_prototypes import main
        main()
