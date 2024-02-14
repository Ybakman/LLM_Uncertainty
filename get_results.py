import argparse
import json
import pickle

import config
import numpy as np
import pandas as pd
import sklearn
import sklearn.metrics
import torch
import wandb

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='facebook/opt-350m')
parser.add_argument('--run_id', type=str, default='run_1')
args = parser.parse_args()

model_name = args.model_name.replace("/", "")

with open(f'{config.output_dir}/sequences/{args.run_id}/{model_name}_generations_similarities.pkl', 'rb') as f:
    similarities_dict = pickle.load(f)

with open(f'{config.output_dir}/sequences/{args.run_id}/{model_name}_generations.pkl', 'rb') as infile:
    cleaned_sequences = pickle.load(infile)

with open(f'{config.output_dir}/sequences/{args.run_id}/aggregated_likelihoods_{model_name}_generations.pkl', 'rb') as f:
    overall_results  = pickle.load(f)


wandb.init(project='mars', id=args.run_id, config=args, resume='allow')
run_name = wandb.run.name


similarities_df = pd.DataFrame.from_dict(similarities_dict, orient='index')
similarities_df['id'] = similarities_df.index
similarities_df['has_semantically_different_answers'] = similarities_df[
    'has_semantically_different_answers'].astype('int')


generations_df = pd.DataFrame(cleaned_sequences)
generations_df['id'] = generations_df['id'].apply(lambda x: x[0])
generations_df['id'] = generations_df['id'].astype('object')

generations_df['length_of_most_likely_generation'] = generations_df['most_likely_generation'].apply(
    lambda x: len(str(x).split(' ')))

#generations_df['length_of_answer'] = generations_df['answer'].apply(lambda x: len(str(x).split(' ')))

generations_df['variance_of_length_of_generations'] = generations_df['generated_texts'].apply(
    lambda x: np.var([len(str(y).split(' ')) for y in x]))



generations_df['correct'] = (generations_df['gpt_score'] > 90).astype('int')

num_generations = len(generations_df['generated_texts'][0])
likelihoods = overall_results

keys_to_use = ('ids', 'predictive_entropy', 'mutual_information', 'average_predictive_entropy',\
                'average_pointwise_mutual_information', 'average_neg_log_likelihood_of_most_likely_gen',\
                 'neg_log_likelihood_of_most_likely_gen',\
                'predictive_entropy_over_concepts', 'number_of_semantic_sets', 'unnormalised_entropy_over_concepts',\
                'scores_importance_mean','scores_importance_max','scores_importance_min','scores_prob',\
                'predictive_entropy_over_concepts_importance_mean','predictive_entropy_over_concepts_importance_max','predictive_entropy_over_concepts_importance_min'\
                , 'average_predictive_entropy_importance_mean', 'average_predictive_entropy_importance_max', 'average_predictive_entropy_importance_min')

likelihoods_small = dict((k, likelihoods[k]) for k in keys_to_use)
for key in likelihoods_small:
    if key == 'average_predictive_entropy_on_subsets':
        likelihoods_small[key].shape
    if type(likelihoods_small[key]) is torch.Tensor:
        likelihoods_small[key] = torch.squeeze(likelihoods_small[key].cpu())

likelihoods_df = pd.DataFrame.from_dict(likelihoods_small)
likelihoods_df.rename(columns={'ids': 'id'}, inplace=True)

result_df = generations_df.merge(similarities_df, on='id').merge(likelihoods_df, on='id')

n_samples_before_filtering = len(result_df)
result_df['len_most_likely_generation_length'] = result_df['most_likely_generation'].apply(lambda x: len(x.split()))

overall_result_dict = {}
# Begin analysis
result_dict = {}
result_dict['accuracy'] = result_df['correct'].mean()

print(result_dict['accuracy'])

# Compute the auroc for the length normalized predictive entropy
ln_predictive_entropy_auroc = sklearn.metrics.roc_auc_score(1 - result_df['correct'],
                                                            result_df['average_predictive_entropy'])#bug

result_dict['ln_predictive_entropy_auroc'] = ln_predictive_entropy_auroc

ln_predictive_entropy_auroc_importance_mean = sklearn.metrics.roc_auc_score(1 - result_df['correct'],
                                                            result_df['average_predictive_entropy_importance_mean'])
result_dict['ln_predictive_entropy_auroc_importance_mean'] = ln_predictive_entropy_auroc_importance_mean

ln_predictive_entropy_auroc_importance_max = sklearn.metrics.roc_auc_score(1 - result_df['correct'],
                                                            result_df['average_predictive_entropy_importance_max'])
result_dict['ln_predictive_entropy_auroc_importance_max'] = ln_predictive_entropy_auroc_importance_max

ln_predictive_entropy_auroc_importance_min = sklearn.metrics.roc_auc_score(1 - result_df['correct'],
                                                            result_df['average_predictive_entropy_importance_min'])
result_dict['ln_predictive_entropy_auroc_importance_min'] = ln_predictive_entropy_auroc_importance_min


predictive_entropy_auroc = sklearn.metrics.roc_auc_score(1 - result_df['correct'], result_df['predictive_entropy'])
result_dict['predictive_entropy_auroc'] = predictive_entropy_auroc

entropy_over_concepts_auroc = sklearn.metrics.roc_auc_score(1 - result_df['correct'],
                                                            result_df['predictive_entropy_over_concepts'])
result_dict['entropy_over_concepts_auroc'] = entropy_over_concepts_auroc

entropy_over_concepts_auroc_importance_mean = sklearn.metrics.roc_auc_score(1 - result_df['correct'],
                                                            result_df['predictive_entropy_over_concepts_importance_mean'])
result_dict['entropy_over_concepts_auroc_importance_mean'] = entropy_over_concepts_auroc_importance_mean

entropy_over_concepts_auroc_importance_max = sklearn.metrics.roc_auc_score(1 - result_df['correct'],
                                                            result_df['predictive_entropy_over_concepts_importance_max'])
result_dict['entropy_over_concepts_auroc_importance_max'] = entropy_over_concepts_auroc_importance_max

entropy_over_concepts_auroc_importance_min = sklearn.metrics.roc_auc_score(1 - result_df['correct'],
                                                            result_df['predictive_entropy_over_concepts_importance_min'])
result_dict['entropy_over_concepts_auroc_importance_min'] = entropy_over_concepts_auroc_importance_min


unnormalised_entropy_over_concepts_auroc = sklearn.metrics.roc_auc_score(1 - result_df['correct'], result_df['unnormalised_entropy_over_concepts'])
result_dict['unnormalised_entropy_over_concepts_auroc'] = unnormalised_entropy_over_concepts_auroc


scores_importance_mean = sklearn.metrics.roc_auc_score(1-result_df['correct'],
                                                            result_df['scores_importance_mean'])
result_dict['scores_importance_mean'] = scores_importance_mean

scores_importance_max = sklearn.metrics.roc_auc_score(1-result_df['correct'],
                                                            result_df['scores_importance_max'])
result_dict['scores_importance_max'] = scores_importance_max

scores_importance_min = sklearn.metrics.roc_auc_score(1-result_df['correct'],
                                                            result_df['scores_importance_min'])
result_dict['scores_importance_min'] = scores_importance_min

scores_prob = sklearn.metrics.roc_auc_score(1-result_df['correct'],
                                                            result_df['scores_prob'])
result_dict['scores_prob'] = scores_prob

neg_llh_most_likely_gen_auroc = sklearn.metrics.roc_auc_score(1 - result_df['correct'],
                                                                result_df['neg_log_likelihood_of_most_likely_gen'])
result_dict['neg_llh_most_likely_gen_auroc'] = neg_llh_most_likely_gen_auroc
number_of_semantic_sets_auroc = sklearn.metrics.roc_auc_score(1 - result_df['correct'],
                                                                result_df['number_of_semantic_sets'])
result_dict['number_of_semantic_sets_auroc'] = number_of_semantic_sets_auroc

result_dict['number_of_semantic_sets_correct'] = result_df[result_df['correct'] ==
                                                            1]['number_of_semantic_sets'].mean()
result_dict['number_of_semantic_sets_incorrect'] = result_df[result_df['correct'] ==
                                                                0]['number_of_semantic_sets'].mean()

average_neg_llh_most_likely_gen_auroc = sklearn.metrics.roc_auc_score(
    1 - result_df['correct'], result_df['average_neg_log_likelihood_of_most_likely_gen'])
result_dict['average_neg_llh_most_likely_gen_auroc'] = average_neg_llh_most_likely_gen_auroc

wandb.log(result_dict)
wandb.finish()
with open(f'result_{args.run_id}.json', 'w') as f:
    json.dump(overall_result_dict, f)

