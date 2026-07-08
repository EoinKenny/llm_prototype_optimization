"""Prompt templates transcribed from Appendix H of the manuscript."""
from __future__ import annotations

import json
from typing import Sequence

import numpy as np


DESCRIPTIONS = {
    "trec": "Question",
    "20newsgroups": "News Article",
    "dbpedia": "Wikipedia Page",
    "imdb": "Movie Review",
    "amazon_reviews": "Amazon Review",
    "agnews": "News Article",
}


def initialization_prompt(
    examples: Sequence[str],
    num_guesses_to_generate: int,
    dataset_name: str,
) -> str:
    description = DESCRIPTIONS[dataset_name]
    examples_str = "\n".join(f'- "{example}"' for example in examples)
    return f"""I am trying to identify a prototypical example from the {dataset_name} dataset.

The prototype should represent a typical example of a '{description}'.
The following examples are very similar to the real prototype:
{examples_str}

Based *only* on these examples, please generate a Python list containing exactly
{num_guesses_to_generate} distinct, concise, and relevant phrases or sentences
that you believe also capture the core concepts in these examples in a
prototypical sentence.

Each phrase should be a potential textual description of the prototype and its
core concepts.

Your output must be ONLY a single Python list of strings. For example:
["first candidate phrase", "second candidate phrase", ..., "tenth candidate phrase"]

Generated Python list:
"""


def optimization_prompt(
    population: Sequence[str],
    distances: Sequence[float],
    num_neighbors: int,
    training_examples: Sequence[str],
    dataset: str,
) -> str:
    description = DESCRIPTIONS[dataset]
    rounded = [round(float(value), 2) for value in np.asarray(distances).flatten()]
    return f"""You are a helpful assistant to a data scientist.

We are working together to try find a text sequence which perfectly maps to a
learned black-box prototype vector in the latent space of a language model.
In doing so, we are querying you repeatedly in an optimization loop.
This is one of those loops.

I will show you the current {num_neighbors} text sequences you generated
previously, and their cosine similarity to the prototype.
The closer the similarity is to 1, the better the guess is, because it's more
similar to the prototype; the similarity ranges from -1 to 1.
Our goal is to find a text sequence which perfectly maps to the prototype and
gives a score of 1.

Here are the current {num_neighbors} text sequences you have generated
previously in a query: {list(population)}
Their similarity scores are:
{rounded}

Can you suggest another {num_neighbors} guesses which are closer to 1?

The prototype should represent a short, prototypical example of a
'{description}'.

If a lot of your guesses are similar, you should try to diversify them to avoid
getting stuck in a local minimum; you can try varying the length, or even take
random guesses.
Here are some close training data neighbors of the black-box prototype to help
you get some variety in your guesses: {list(training_examples)}

Respond ONLY with your guesses as a Python list of strings.

For example:

["first guess",
"second guess",
"...",
"last guess"]

It is extremely important you follow this format exactly.
"""


def qualitative_prompt(dataset: str, test_text: str, stage_a_proto: str, stage_b_proto: str) -> str:
    return f"""You are analyzing prototypes used by a neural network classifier that uses
cosine similarity for classification on the {dataset} dataset. The prototypes
are being used to classify the test instance based on their cosine similarity
to it; your job is to help us analyze if the prototypes have meaningful
similarity to the test instance.
## Test Instance to Classify:
{test_text}
## Stage A Prototype:
{stage_a_proto}
## Stage B Prototype:
{stage_b_proto}
Please analyze these prototypes and the test instance:
1. First, identify ALL high-level concepts in the Stage A prototype that could
   be used by a classifier.
2. Do the same analysis for the Stage B prototype.
3. Analyze which concepts from each prototype are actually present in the test
   instance.
4. Identify any overly specific, or irrelevant features in each prototype that
   might mislead the classifier, and would not easily generalize to other
   instances of the class.
5. Based on cosine similarity principles, determine which prototype would be
   most similar to the test instance.
You should be comprehensive; you don't need to have the same number of concepts
for both prototypes, it is ok for one to have many more concepts and/or
irrelevant features.
Provide detailed reasoning for your analysis, then output a JSON object with the
following structure:
```json
{{
  "stage_a_concepts_count": <integer>,
  "stage_b_concepts_count": <integer>,
  "stage_a_concepts_in_test": <integer>,
  "stage_b_concepts_in_test": <integer>,
  "stage_a_irrelevant_features": <integer>,
  "stage_b_irrelevant_features": <integer>,
  "most_similar_prototype": "<'stage_a' or 'stage_b'>",
  "confidence": <float between 0 and 1>
}}
```
The length and detail level of the prototypes do NOT matter for classification
purposes; do not consider them in your analysis, only focus on high-level
concepts for the classification.
"""
