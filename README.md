# Rethinking Prototype Projection — Reproduction Code

This repository trains the prototype classifiers, performs nearest-neighbor and LLM-generated projection, and reproduces the quantitative and qualitative analyses in the paper.

## 1. Environment

The full pipeline is intended for Linux with four NVIDIA CUDA GPUs and Python 3.11.

```bash
conda create -n prototype-projection python=3.11 pip -y
conda activate prototype-projection

# Reference CUDA 12.4 installation
python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
```

Check that CUDA is available:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

## 2. Download the datasets

Install and configure the Kaggle command-line tool, then run:

```bash
mkdir -p datasets/{20newsgroups,agnews,amazon_reviews,dbpedia,imdb,trec}

kaggle datasets download -d ducanger/imdb-dataset \
  -p datasets/imdb
kaggle datasets download -d amananandrai/ag-news-classification-dataset \
  -p datasets/agnews
kaggle datasets download -d abdallahwagih/amazon-reviews \
  -p datasets/amazon_reviews
kaggle datasets download -d danofer/dbpedia-classes \
  -p datasets/dbpedia
kaggle datasets download \
  -d thedevastator/the-trec-question-classification-dataset-a-longi \
  -p datasets/trec
```

The pipeline downloads 20 Newsgroups through scikit-learn and automatically extracts the Kaggle archives.

## 3. Configure the models

Accept access to `meta-llama/Meta-Llama-3-8B-Instruct` on Hugging Face, then set:

```bash
export HF_TOKEN="your_token"
export PAPER_DEVICES="cuda:0,cuda:1,cuda:2,cuda:3"
export OPTIMIZER_GPU_IDS="1,2,3"
```

For the five qualitative judges, implement this function in `src/judge_clients.py`:

```python
def call_judge_api(
    prompts: list[str],
    provider: str,
    model_id: str,
) -> list[str]:
    ...
```

It must return one response string per prompt, in the same order. The judge names, providers, and model IDs are defined in `src/config.py`.

## 4. Run the full pipeline

From the repository root:

```bash
python reproduce_paper.py
```

This sequentially preprocesses the data, trains all models, optimizes and projects the prototypes, builds the qualitative prompts, runs the five judges, and produces the final analyses.

Existing checkpoints and completed judge responses are reused when the command is restarted.

Outputs are written to:

```text
weights/
results/optimization/
results/qualitative/
results/analysis/
```

Individual stages can also be run with `run0_prepare_data.py` through `run7_analyze_qualitative.py`.
