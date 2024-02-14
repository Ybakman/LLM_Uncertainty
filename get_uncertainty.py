import argparse
import os
import pickle
import random

import config
import numpy as np
import torch
import wandb
import copy
import pandas as pd
import sklearn
import sklearn.metrics
from sentence_transformers import SentenceTransformer 
import IPython

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='facebook/opt-350m')
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
wandb.init(project='llm-uncertainty',entity='yavuz-team', id=args.run_id, config=args, resume='allow')
run_name = wandb.run.name

llh_shift = torch.tensor(5.0)#does not effect anything

model_name = args.model_name.replace("/", "")


with open(f'{config.data_dir}/sequences/{run_name}/{model_name}_generations_likelihoods.pkl', 'rb') as infile:
    result = pickle.load(infile)

with open(f'{config.output_dir}/sequences/{run_name}/{model_name}_generations_similarities.pkl', 'rb') as infile:
    similarities_dict = pickle.load(infile)

with open(f'{config.output_dir}/sequences/{run_name}/{model_name}_generations.pkl', 'rb') as infile:
    cleaned_sequences = pickle.load(infile)


def get_overall_log_likelihoods(list_of_results):
    """Compute log likelihood of all generations under their given context.
    
    list_of_results: list of dictionaries with keys:
    
    returns: dictionary with keys: 'neg_log_likelihoods', 'average_neg_log_likelihoods'
             that contains tensors of shape (num_models, num_generations, num_samples_per_generation)
    """

    result_dict = {}
    geometric_dict ={}

    list_of_keys = ['neg_log_likelihoods', 'average_neg_log_likelihoods',\
                    'pointwise_mutual_information', 'average_neg_log_likelihood_of_most_likely_gen',\
                    'neg_log_likelihood_of_most_likely_gen', 'semantic_set_ids', \
                    'average_neg_log_likelihoods_importance_mean', 'average_neg_log_likelihoods_importance_max', 'average_neg_log_likelihoods_importance_min',\
                    'most_likely_neg_log_likelihoods', 
                    'most_likely_neg_log_likelihoods_importance_mean', 'most_likely_neg_log_likelihoods_importance_max', 'most_likely_neg_log_likelihoods_importance_min']

    geometric_keys = ['has_different_answers','unique_answers_indices']

    for key in geometric_keys:
        overall_results = []
        for sample in list_of_results:
            overall_results.append(sample[key])
        geometric_dict[key]  = overall_results

    for key in list_of_keys:
        list_of_ids = []
        overall_results = []
        results_per_model = []
        for sample in list_of_results:
            average_neg_log_likelihoods = sample[key]
            list_of_ids.append(sample['id'][0])
            results_per_model.append(average_neg_log_likelihoods)

        results_per_model = torch.stack(results_per_model)

        overall_results.append(results_per_model)

        if key not in ['meaning_vectors', 'meaning_vectors_only_answer','has_different_answers']:
            overall_results = torch.stack(overall_results)

        result_dict[key] = overall_results

    result_dict['ids'] = list_of_ids
    return result_dict,geometric_dict


def get_mutual_information(log_likelihoods):
    """Compute confidence measure for a given set of likelihoods"""

    mean_across_models = torch.logsumexp(log_likelihoods, dim=0) - torch.log(torch.tensor(log_likelihoods.shape[0]))
    tiled_mean = mean_across_models.tile(log_likelihoods.shape[0], 1, 1)
    diff_term = torch.exp(log_likelihoods) * log_likelihoods - torch.exp(tiled_mean) * tiled_mean
    f_j = torch.div(torch.sum(diff_term, dim=0), diff_term.shape[0])
    mutual_information = torch.div(torch.sum(torch.div(f_j, mean_across_models), dim=1), f_j.shape[-1])

    return mutual_information


def get_log_likelihood_variance(neg_log_likelihoods):
    """Compute log likelihood variance of approximate posterior predictive"""
    mean_across_models = torch.mean(neg_log_likelihoods, dim=0)
    variance_of_neg_log_likelihoods = torch.var(mean_across_models, dim=1)

    return variance_of_neg_log_likelihoods


def get_log_likelihood_mean(neg_log_likelihoods):
    """Compute softmax variance of approximate posterior predictive"""
    mean_across_models = torch.mean(neg_log_likelihoods, dim=0)
    mean_of_neg_log_likelihoods = torch.mean(mean_across_models, dim=1)

    return mean_of_neg_log_likelihoods


def get_mean_of_poinwise_mutual_information(pointwise_mutual_information):
    """Compute mean of pointwise mutual information"""
    mean_across_models = torch.mean(pointwise_mutual_information, dim=0)
    return torch.mean(mean_across_models, dim=1)


def get_predictive_entropy(log_likelihoods):
    """Compute predictive entropy of approximate posterior predictive"""
    #log_likelihoods = log_likelihoods[:,:,:1]
    #print(log_likelihoods.shape)
    mean_across_models = torch.logsumexp(log_likelihoods, dim=0) - torch.log(torch.tensor(log_likelihoods.shape[0]))
    entropy = -torch.sum(mean_across_models, dim=1) / torch.tensor(mean_across_models.shape[1])
    return entropy


def get_predictive_entropy_over_concepts(log_likelihoods, semantic_set_ids):
    """Compute the semantic entropy"""
    #log_likelihoods = log_likelihoods[:,:,:1]
    mean_across_models = torch.logsumexp(log_likelihoods, dim=0) - torch.log(torch.tensor(log_likelihoods.shape[0]))
    # This is ok because all the models have the same semantic set ids
    semantic_set_ids = semantic_set_ids[0]
    entropies = []
    for row_index in range(mean_across_models.shape[0]):
        aggregated_likelihoods = []
        row = mean_across_models[row_index]
        semantic_set_ids_row = semantic_set_ids[row_index]
        #semantic_set_ids_row = semantic_set_ids_row[:1]
        for semantic_set_id in torch.unique(semantic_set_ids_row):
            aggregated_likelihoods.append(torch.logsumexp(row[semantic_set_ids_row == semantic_set_id], dim=0))
        aggregated_likelihoods = torch.tensor(aggregated_likelihoods) - llh_shift
        entropy = - torch.sum(aggregated_likelihoods, dim=0) / torch.tensor(aggregated_likelihoods.shape[0])
        entropies.append(entropy)

    return torch.tensor(entropies)


def get_margin_probability_uncertainty_measure(log_likelihoods):
    """Compute margin probability uncertainty measure"""
    mean_across_models = torch.logsumexp(log_likelihoods, dim=0) - torch.log(torch.tensor(log_likelihoods.shape[0]))
    topk_likelihoods, indices = torch.topk(mean_across_models, 2, dim=1, sorted=True)
    margin_probabilities = np.exp(topk_likelihoods[:, 0]) - np.exp(topk_likelihoods[:, 1])

    return margin_probabilities


overall_results,geometric_results = get_overall_log_likelihoods(result)



average_pointwise_mutual_information = get_mean_of_poinwise_mutual_information(
    overall_results['pointwise_mutual_information'])

mutual_information = get_mutual_information(overall_results['neg_log_likelihoods'])
predictive_entropy = get_predictive_entropy(-overall_results['neg_log_likelihoods'])
predictive_entropy_over_concepts = get_predictive_entropy_over_concepts(-overall_results['average_neg_log_likelihoods'],
                                                                        overall_results['semantic_set_ids'])


unnormalised_entropy_over_concepts = get_predictive_entropy_over_concepts(-overall_results['neg_log_likelihoods'],
                                                                          overall_results['semantic_set_ids'])#proposed algorithm
margin_measures = get_margin_probability_uncertainty_measure(-overall_results['average_neg_log_likelihoods'])
unnormalised_margin_measures = get_margin_probability_uncertainty_measure(-overall_results['neg_log_likelihoods'])

scores_prob = overall_results['most_likely_neg_log_likelihoods']

scores_importance_mean = overall_results['most_likely_neg_log_likelihoods_importance_mean']
scores_importance_max = overall_results['most_likely_neg_log_likelihoods_importance_max']
scores_importance_min = overall_results['most_likely_neg_log_likelihoods_importance_min']



predictive_entropy_over_concepts_importance_mean = get_predictive_entropy_over_concepts(-overall_results['average_neg_log_likelihoods_importance_mean'],
                                                                        overall_results['semantic_set_ids'])    
predictive_entropy_over_concepts_importance_max = get_predictive_entropy_over_concepts(-overall_results['average_neg_log_likelihoods_importance_max'],
                                                                        overall_results['semantic_set_ids'])    
predictive_entropy_over_concepts_importance_min = get_predictive_entropy_over_concepts(-overall_results['average_neg_log_likelihoods_importance_min'],
                                                                        overall_results['semantic_set_ids']) 

def get_number_of_unique_elements_per_row(tensor):
    assert len(tensor.shape) == 2
    return torch.count_nonzero(torch.sum(torch.nn.functional.one_hot(tensor), dim=1), dim=1)

number_of_semantic_sets = get_number_of_unique_elements_per_row(overall_results['semantic_set_ids'][0])
average_predictive_entropy = get_predictive_entropy(-overall_results['average_neg_log_likelihoods'])


average_predictive_entropy_importance_mean = get_predictive_entropy(-overall_results['average_neg_log_likelihoods_importance_mean'])
average_predictive_entropy_importance_max = get_predictive_entropy(-overall_results['average_neg_log_likelihoods_importance_max'])
average_predictive_entropy_importance_min = get_predictive_entropy(-overall_results['average_neg_log_likelihoods_importance_min'])


overall_results['mutual_information'] = mutual_information
overall_results['predictive_entropy'] = predictive_entropy
overall_results['predictive_entropy_over_concepts'] = predictive_entropy_over_concepts
overall_results['unnormalised_entropy_over_concepts'] = unnormalised_entropy_over_concepts
overall_results['number_of_semantic_sets'] = number_of_semantic_sets
overall_results['margin_measures'] = margin_measures
overall_results['unnormalised_margin_measures'] = unnormalised_margin_measures


overall_results['scores_prob'] = scores_prob
overall_results['scores_importance_mean'] = scores_importance_mean
overall_results['scores_importance_max'] = scores_importance_max
overall_results['scores_importance_min'] = scores_importance_min

overall_results['average_predictive_entropy'] = average_predictive_entropy
overall_results['average_pointwise_mutual_information'] = average_pointwise_mutual_information


overall_results['average_predictive_entropy_importance_mean'] = average_predictive_entropy_importance_mean
overall_results['average_predictive_entropy_importance_max'] = average_predictive_entropy_importance_max
overall_results['average_predictive_entropy_importance_min'] = average_predictive_entropy_importance_min

overall_results['predictive_entropy_over_concepts_importance_mean'] = predictive_entropy_over_concepts_importance_mean
overall_results['predictive_entropy_over_concepts_importance_max'] = predictive_entropy_over_concepts_importance_max
overall_results['predictive_entropy_over_concepts_importance_min'] = predictive_entropy_over_concepts_importance_min


with open(f'{config.output_dir}/sequences/{run_name}/aggregated_likelihoods_{model_name}_generations.pkl',
          'wb') as outfile:
    pickle.dump(overall_results, outfile)
