"""Sequential entry point for the judges stage."""
from src.toy_pipeline import activate_from_argv, run_stage

if __name__ == "__main__":
    if activate_from_argv():
        run_stage("judges")
    else:
        from src.run_judges import main
        main()
