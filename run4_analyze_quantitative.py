"""Sequential entry point for the quantitative analysis stage."""
from src.toy_pipeline import activate_from_argv, run_stage

if __name__ == "__main__":
    if activate_from_argv():
        run_stage("quantitative_analysis")
    else:
        from src.analyze_quantitative import main
        main()
