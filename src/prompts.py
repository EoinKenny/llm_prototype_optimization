import numpy as np


def make_prompt(population, distances, num_neighbors, training_examples=None, dataset=None, class_desc=None):
    """
    Construct the prompt for LLM optimization
    You can pass training examples if desired to help diversify responses and guess from LLM, although in practice this seems to hurt.
    """
    
    if dataset=='trec':
        description = 'Question'
    elif dataset == '20newsgroups':
        description = 'News Article'
    elif dataset == 'dbpedia':
        description = 'Wikipedia Page'
    else:
        raise NameError('wrong dataset name')    
    
    prompt = f"""
    You are a helpful assistant to a data scientist.

    We are working together to try find a text sequence which perfectly maps to a learned black box prototype vector in the latent space of a language model.
    In doing so, we are querying you repeatedly in an optimization loop.
    This is one of those loops.

    I will show you the current {num_neighbors} text sequences you generated perviously, and their similarity to the prototype.
    The closer the similarity is to 1, the better the guess is, because it's more similar to the prototype, the similarity ranges from 0 to 1.
    Our goal is to find a text sequence which perfectly maps to the prototype and gives a score of 1.

    Here are the current {num_neighbors} text sequences you have generated previously in a query: {population}
    Their similarity scores are: {[round(c, 2) for c in np.array(distances).flatten()]}

    Can you suggest another {num_neighbors} guesses which are closer to 1? 

    The prototype should represent a short prototypical example of a '{description}'. 
    Your guesses should focus on core concepts in the domain.
    
    If a lot of your guesses are similar, you should try diversify them to avoid getting stuck in a local minimum, you can try vary the length, or even take random guesses.
    Here are some close neighbors of the black-box prototype to help you make better guesses: {training_examples}

    Outline your reasoning and then finally state your guesses as a Python list of strings. For example: 
    
    ["first guess", 
    "second guess", 
    "...",
    "last guess"]

    It is extremely important you follow this format exactly. Do not write a bullet point list.
    """
        
    return prompt

