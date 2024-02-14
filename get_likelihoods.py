import argparse
import os
import pathlib
import pickle
from lib2to3.pgen2.tokenize import tokenize

import accelerate
import config
import datasets
import evaluate
import numpy as np
import torch
from tqdm import tqdm
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer,AutoModelForSequenceClassification
import os
import random
import csv
import copy
import pandas as pd
import sklearn
import sklearn.metrics
from sentence_transformers import SentenceTransformer 
import IPython

import nltk
from nltk.tokenize import word_tokenize

from flair.data import Sentence
from flair.models import SequenceTagger


import wandb
import re

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='facebook/opt-350m')
parser.add_argument('--run_id', type=str, default='run_1')
parser.add_argument('--device', type=int, default=0)
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

device = torch.device("cuda:" + str(args.device) if torch.cuda.is_available() else "cpu")

model = AutoModelForCausalLM.from_pretrained(f"{args.model_name}",
                                             torch_dtype=torch.float16).to(device)

tokenizer = AutoTokenizer.from_pretrained(f"{args.model_name}",
                                          use_fast=False)

wandb.init(project='mars', id=args.run_id, config=args, resume='allow')

run_name = wandb.run.name

model_name = args.model_name.replace("/", "")

with open(f'{config.output_dir}/sequences/{run_name}/{model_name}_generations.pkl', 'rb') as infile:
    sequences = pickle.load(infile)

with open(f'{config.output_dir}/sequences/{run_name}/{model_name}_generations_similarities.pkl', 'rb') as infile:
    similarities_dict = pickle.load(infile)


def softmax_with_temperature(logits, temperature):
    scaled_logits = logits / temperature
    softmax_probs = torch.nn.functional.softmax(scaled_logits, dim=0)
    return softmax_probs

IGNORE_INDEX = -100
def compute_token_nll(model_output, prompt_len, generation):
    # log probabilities of the target words
    # Just in case the loss is not NLL for the model
    #assert len(generation.shape) == 1
    _logits = model_output['logits'][0, prompt_len-1:-1]
    criterion = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, reduction='mean')
    loss = criterion(_logits, generation[prompt_len:])
    if torch.isnan(loss):
        loss = 100000
    return loss

def compute_token_nll_importance_phrase(model_output, prompt_len, generation, importance_vector,phrases, mode = 'mean'):
    importance_vector = importance_vector.to(device)
    #make the normalization
    
    #assert len(generation.shape) == 1
    _logits = model_output['logits'][0, prompt_len-1:-1]


    criterion = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, reduction='mean')
    #assert generation[prompt_len:].ne(IGNORE_INDEX).all()

    _logits = _logits.float()

    ids = generation[prompt_len:]

    probs = torch.nn.functional.softmax(_logits, dim=1)
    probs = torch.gather(probs, dim=1, index=ids.view(-1, 1))
    neg_log_likelihoods = -torch.log( probs.reshape(-1)) 
    neg_log_likelihoods = neg_log_likelihoods.reshape(-1)
    #find probabilities of each word
    neg_log_likelihoods_word = []
    token_idx = 0
    merged_importance_vector  = []
    i= 0
    #print(ids)
    while i < len(phrases):
        found = False
        while found == False:
            for k in range(1,len(phrases)-i+1):
                word  = "".join(phrases[i:i+k])
                #print(word)
                last_token = -1
                for j in range(token_idx+1, len(ids)+1):#importance should be summed I guess
                    if tokenizer.decode(ids[token_idx:j]).strip().replace(" ", "").lower() == word.strip().replace(" ", "").lower():
                        last_token = j
                    
                if last_token != -1:
                    if mode == 'mean':
                        neg_log_likelihoods_word.append(torch.mean(neg_log_likelihoods[token_idx:last_token]))
                        merged_importance_vector.append(torch.mean(importance_vector[i:i+k]))
                    elif mode == 'max':
                        neg_log_likelihoods_word.append(torch.max(neg_log_likelihoods[token_idx:last_token]))
                        merged_importance_vector.append(torch.mean(importance_vector[i:i+k]))
                    elif mode == 'min':
                        neg_log_likelihoods_word.append(torch.min(neg_log_likelihoods[token_idx:last_token]))
                        merged_importance_vector.append(torch.mean(importance_vector[i:i+k]))
                    found = True
                    i += k
                    token_idx = last_token 
                    break
        #print(i)
    
    neg_log_likelihoods_word = torch.tensor(neg_log_likelihoods_word).to(device)
    merged_importance_vector = torch.tensor(merged_importance_vector).to(device)
    merged_importance_vector = merged_importance_vector/torch.sum(merged_importance_vector)

    if 'medical' in args.run_id:
        merged_importance_vector = softmax_with_temperature(merged_importance_vector,0.001)#only for medical dataset
        score = 0.5 * torch.sum( merged_importance_vector * neg_log_likelihoods_word) + 0.5 * torch.mean(neg_log_likelihoods )
    else:
        score = 0.5 * torch.sum( merged_importance_vector * neg_log_likelihoods_word) + 0.5 * torch.mean(neg_log_likelihoods )

    if torch.isnan(score):
        score = 100000
    return score

  
tokenizer.pad_token_id = 1#very crucial don't forget


result = []
model.eval()
with torch.no_grad():
    for i,sample in tqdm(enumerate(sequences)):
        result_dict = {}
        prompt = sample['prompt']

        generations = sample['cleaned_generations'].to(device)
        
        id_ = sample['id']

        importance_vector_most_likely = similarities_dict[id_[0]]['importance_vector'][0]
        phrases_most_likely = similarities_dict[id_[0]]['importance_vector'][1]
        importance_scores = similarities_dict[id_[0]]['importance_scores']


        average_neg_log_likelihoods = torch.zeros((generations.shape[0],))
        average_neg_log_likelihoods_importance_mean = torch.zeros((generations.shape[0],))
        average_neg_log_likelihoods_importance_max = torch.zeros((generations.shape[0],))
        average_neg_log_likelihoods_importance_min = torch.zeros((generations.shape[0],))
        
        average_unconditioned_neg_log_likelihoods = torch.zeros((generations.shape[0],))
        neg_log_likelihoods = torch.zeros((generations.shape[0],))
        neg_unconditioned_log_likelihoods = torch.zeros((generations.shape[0],))
        pointwise_mutual_information = torch.zeros((generations.shape[0],))
        sequence_embeddings = []

        for generation_index in range(generations.shape[0]):#
            prompt = prompt[prompt != tokenizer.pad_token_id]
            generation = generations[generation_index][generations[generation_index] != tokenizer.pad_token_id] # generation includes prompt

            # This computation of the negative log likelihoods follows this tutorial: https://huggingface.co/docs/transformers/perplexity
            target_ids = generation.clone()
            #print(len(generation)-len(prompt))
            target_ids[:len(prompt)] = -100
            model_output = model(torch.reshape(generation, (1, -1)), labels=target_ids, output_hidden_states=False)
            generation_only = generation.clone()[(len(prompt) - 1):]
            unconditioned_model_output = model(torch.reshape(generation_only, (1, -1)),
                                                labels=generation_only,
                                                output_hidden_states=False)#ignore prompt to get unconditioned model output

            model_output_loss = compute_token_nll(model_output, len(prompt), generation.reshape(-1))#using this is more safe
            unconditioned_model_output_loss = compute_token_nll(unconditioned_model_output, 1, generation_only.reshape(-1))#using this is more safe

            importance_score = importance_scores[generation_index][0]
            phrases  = importance_scores[generation_index][1]

            model_output_loss_importance_mean = compute_token_nll_importance_phrase(model_output, len(prompt), generation.reshape(-1), importance_score,phrases,mode='mean')
            model_output_loss_importance_max = compute_token_nll_importance_phrase(model_output, len(prompt), generation.reshape(-1), importance_score,phrases,mode='max')
            model_output_loss_importance_min = compute_token_nll_importance_phrase(model_output, len(prompt), generation.reshape(-1), importance_score,phrases,mode='min')

            average_neg_log_likelihoods_importance_mean[generation_index] = model_output_loss_importance_mean
            average_neg_log_likelihoods_importance_max[generation_index] = model_output_loss_importance_max
            average_neg_log_likelihoods_importance_min[generation_index] = model_output_loss_importance_min
            
            average_neg_log_likelihood = model_output_loss
     
            average_unconditioned_neg_log_likelihood = unconditioned_model_output_loss
            average_neg_log_likelihoods[generation_index] = average_neg_log_likelihood
            average_unconditioned_neg_log_likelihoods[generation_index] = average_unconditioned_neg_log_likelihood

            # total neg lok likelihoods
            neg_log_likelihoods[generation_index] = average_neg_log_likelihood * (len(generation) - len(prompt))
            neg_unconditioned_log_likelihoods[generation_index] = average_unconditioned_neg_log_likelihood * (
                len(generation) - len(prompt))

            pointwise_mutual_information[generation_index] = -neg_log_likelihoods[
                generation_index] + neg_unconditioned_log_likelihoods[generation_index]


        
        # do the same thing above to first and second most likely generations
        most_likely_generation = sample['cleaned_most_likely_generation_ids'].to(device)
        most_likely_generation = most_likely_generation[most_likely_generation != tokenizer.pad_token_id]
        
        target_ids = most_likely_generation.clone()
        target_ids[:len(prompt)] = -100
        
        model_output = model(torch.reshape(most_likely_generation, (1, -1)),
                                labels=target_ids,
                                output_hidden_states=False)

        logits = model_output['logits'].cpu()
        model_output_loss = compute_token_nll(model_output, len(prompt), target_ids.reshape(-1))#using this is more safe


        model_output_loss_importance_mean = compute_token_nll_importance_phrase(model_output, len(prompt), target_ids.reshape(-1),importance_vector_most_likely, phrases_most_likely,mode='mean')
        model_output_loss_importance_max = compute_token_nll_importance_phrase(model_output, len(prompt), target_ids.reshape(-1),importance_vector_most_likely, phrases_most_likely, mode='max')
        model_output_loss_importance_min = compute_token_nll_importance_phrase(model_output, len(prompt), target_ids.reshape(-1),importance_vector_most_likely, phrases_most_likely, mode='min')

        result_dict['most_likely_neg_log_likelihoods'] = model_output_loss.cpu()

        result_dict['most_likely_neg_log_likelihoods_importance_mean'] = model_output_loss_importance_mean.cpu()
        result_dict['most_likely_neg_log_likelihoods_importance_max'] = model_output_loss_importance_max.cpu()
        result_dict['most_likely_neg_log_likelihoods_importance_min'] = model_output_loss_importance_min.cpu()

        average_neg_log_likelihood_of_most_likely_gen = model_output_loss

        second_most_likely_generation = sample['second_most_likely_generation_ids'].to(device)
        target_ids = second_most_likely_generation.clone()
        target_ids[:len(prompt)] = -100
    
        model_output = model(torch.reshape(second_most_likely_generation, (1, -1)),
                                labels=target_ids,
                                output_hidden_states=False)


        average_neg_log_likelihood_of_second_most_likely_gen = model_output['loss']

        neg_log_likelihood_of_most_likely_gen = average_neg_log_likelihood_of_most_likely_gen * (
            len(most_likely_generation) - len(prompt))

        
        #sequence_embeddings = torch.stack(sequence_embeddings)
        result_dict['prompt'] = prompt.cpu()
        result_dict['generations'] = generations.cpu()
        result_dict['average_neg_log_likelihoods'] = average_neg_log_likelihoods.cpu()

        result_dict['average_neg_log_likelihoods_importance_mean'] = average_neg_log_likelihoods_importance_mean.cpu()
        result_dict['average_neg_log_likelihoods_importance_max'] = average_neg_log_likelihoods_importance_max.cpu()
        result_dict['average_neg_log_likelihoods_importance_min'] = average_neg_log_likelihoods_importance_min.cpu()
        
        result_dict['neg_log_likelihoods'] = neg_log_likelihoods.cpu()

        result_dict['average_unconditioned_neg_log_likelihoods'] = average_unconditioned_neg_log_likelihoods.cpu()
        result_dict['neg_unconditioned_log_likelihoods'] = neg_unconditioned_log_likelihoods.cpu()
        result_dict['pointwise_mutual_information'] = pointwise_mutual_information.cpu()
        result_dict['average_neg_log_likelihood_of_most_likely_gen'] = average_neg_log_likelihood_of_most_likely_gen.cpu()
        result_dict[
            'average_neg_log_likelihood_of_second_most_likely_gen'] = average_neg_log_likelihood_of_second_most_likely_gen.cpu()
        result_dict['neg_log_likelihood_of_most_likely_gen'] = neg_log_likelihood_of_most_likely_gen.cpu()
        result_dict['semantic_set_ids'] = torch.tensor(similarities_dict[id_[0]]['semantic_set_ids'], device='cpu')
        result_dict['id'] = id_
        
        result_dict['has_different_answers'] = similarities_dict[id_[0]]['has_different_answers']
        result_dict['unique_answers_indices'] = similarities_dict[id_[0]]['unique_answers_indices']


        result.append(result_dict)


with open(f'{config.data_dir}/sequences/{run_name}/{model_name}_generations_likelihoods.pkl',
          'wb') as outfile:
    pickle.dump(result, outfile)