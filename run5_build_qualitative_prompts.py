"""Sequential entry point for the qualitative prompts stage."""
from src.toy_pipeline import activate_from_argv, run_stage

if __name__ == "__main__":
    if activate_from_argv():
        run_stage("qualitative_prompts")
    else:
        from src.build_qualitative_prompts import main
        main()
