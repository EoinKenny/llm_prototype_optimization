"""Sequential entry point for the cache stage."""
from src.toy_pipeline import activate_from_argv, run_stage

if __name__ == "__main__":
    if activate_from_argv():
        run_stage("cache")
    else:
        from src.cache_all import main
        main()
