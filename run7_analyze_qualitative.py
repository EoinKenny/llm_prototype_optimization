"""Sequential entry point for the qualitative analysis stage."""
from src.toy_pipeline import activate_from_argv, run_stage

if __name__ == "__main__":
    if activate_from_argv():
        run_stage("qualitative_analysis")
    else:
        from src.analyze_qualitative import main
        main()
