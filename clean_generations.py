import argparse
import os
import pickle
import random

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

import config
import re

def is_all_valid_chars(s):
    # Pattern includes letters, digits, common punctuation, and whitespace
    pattern = r'^[A-Za-z0-9\s.,;:\'"?!\-_(){}\[\]@#$%^&*+=/|\\<>\~`]+$'
    return bool(re.match(pattern, s))


parser = argparse.ArgumentParser()
parser.add_argument('--generation_model', type=str, default='mistralai/Mistral-7B-v0.1')
parser.add_argument('--run_id', type=str, default='run_1')
args = parser.parse_args()

device = 'cuda'

# Set a seed value
seed_value = 10
# 1. Set `PYTHONHASHSEED` environment variable at a fixed value

os.environ['PYTHONHASHSEED'] = str(seed_value)
# 2. Set `python` built-in pseudo-random generator at a fixed value

random.seed(seed_value)
# 3. Set `numpy` pseudo-random generator at a fixed value

np.random.seed(seed_value)

#Fix torch random seed
torch.manual_seed(seed_value)

os.environ["HF_DATASETS_CACHE"] = config.hf_datasets_cache

generation_tokenizer = AutoTokenizer.from_pretrained(f"{args.generation_model}", use_fast=False, cache_dir=config.data_dir)

run_name = args.run_id

tokenizer = AutoTokenizer.from_pretrained(f"{args.generation_model}", use_fast=False, cache_dir=config.data_dir)

model_name = args.generation_model.replace("/", "")

with open(f'{config.output_dir}/sequences/{run_name}/{model_name}_generations.pkl', 'rb') as infile:
    sequences = pickle.load(infile)

cleaned_sequences = []

for sample in tqdm(sequences):
    discard = False
    cleaned_generations = torch.ones_like(sample['generations'])
    question = sample['question']
    generated_texts = sample['generated_texts']
    cleaned_generated_texts = []

    max_len_of_generations = cleaned_generations.shape[-1]

    strings_to_filter_on = [
        '\n','\xa0', '\x85', '\x87', '\x86', '\x93', '\x94', '\x97', 'Q:', 'A:', 'question:', 'answer:', 'Question:', 'Answer:', 'Questions:', 'questions:', 'QUESTION:',
        'ANSWER:'
    ]

    generated_text = sample['most_likely_generation']
    for string in strings_to_filter_on:
        if string in generated_text:
            generated_text = generated_text.split(string)[0]
       

    if len(generated_text) > 0:
        if tokenizer.__class__.__name__=='PreTrainedTokenizerFast':
            clean_ids = torch.cat(
                    [sample['prompt'].to(device),
                    torch.tensor(tokenizer(generated_text)['input_ids'][0:], device=device)])
        else:
            clean_ids = torch.cat(
                    [sample['prompt'].to(device),
                    torch.tensor(tokenizer(generated_text)['input_ids'][1:], device=device)])

        sample['cleaned_most_likely_generation'] = generated_text
        sample['cleaned_most_likely_generation_ids'] =  clean_ids

        for i, generated_text in enumerate(generated_texts):
            for string in strings_to_filter_on:
                if string in generated_text:
                    generated_text = generated_text.split(string)[0]

            if len(generated_text) == 0:
                discard = True
                break

            cleaned_generated_texts.append(generated_text)
            if tokenizer.__class__.__name__=='PreTrainedTokenizerFast':
                clean_ids = torch.cat(
                    [sample['prompt'].to(device),
                    torch.tensor(tokenizer(generated_text)['input_ids'][0:], device=device)])
            else:
                clean_ids = torch.cat(
                    [sample['prompt'].to(device),
                    torch.tensor(tokenizer(generated_text)['input_ids'][1:], device=device)])
            cleaned_generations[i, :min(len(clean_ids), max_len_of_generations)] = clean_ids[:max_len_of_generations]

        if not discard:
            sample['cleaned_generated_texts'] = cleaned_generated_texts
            sample['cleaned_generations'] = cleaned_generations
            cleaned_sequences.append(sample)

with open(f'{config.output_dir}/sequences/{run_name}/{model_name}_generations.pkl', 'wb') as outfile:
    pickle.dump(cleaned_sequences, outfile)