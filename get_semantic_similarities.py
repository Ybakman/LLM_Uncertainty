import argparse
import csv
import os
import pickle
import random

import sys
print(sys.path)

from lib2to3.pgen2.tokenize import tokenize
import config
import datasets

import evaluate
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sentence_transformers import SentenceTransformer 
from transformers import BertTokenizerFast 
from transformers import DataCollatorForTokenClassification 
from transformers import AutoModelForTokenClassification 


from scipy.special import softmax
import re

import nltk
from nltk.tokenize import word_tokenize

from flair.data import Sentence
from flair.models import SequenceTagger
import unicodedata



parser = argparse.ArgumentParser()
parser.add_argument('--generation_model', type=str, default='facebook/opt-350m')
parser.add_argument('--run_id', type=str, default='run_1')
parser.add_argument('--device', type=int, default=7)

args = parser.parse_args()

device = torch.device("cuda:" + str(args.device) if torch.cuda.is_available() else "cpu")

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

tokenizer = AutoTokenizer.from_pretrained(f"{args.generation_model}", use_fast=False)

run_name = args.run_id

model_name = args.generation_model.replace("/", "")

with open(f'{config.output_dir}/sequences/{run_name}/{model_name}_generations.pkl', 'rb') as infile:
    sequences = pickle.load(infile)


#semantic similarities
model_importance = torch.load('models/model_phrase.pth').to(device)
tokenizer_importance = BertTokenizerFast.from_pretrained("bert-base-uncased") 

semantic_tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-large-mnli")
semantic_model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-large-mnli").to(device)


def inference(model, tokenizer,question, answer):

    words = re.findall(r'\w+|[^\w\s]', answer)
    #first tokenize
    tokenized_input = tokenizer.encode_plus(
    [question],
    words,
    add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
    return_token_type_ids=True,
    is_split_into_words=True,
    truncation=True,
    max_length = 512,           # Pad & truncate all sentences.
    )
    attention_mask = torch.tensor(tokenized_input['attention_mask']).reshape(1,-1).to(device)
    input_ids = torch.tensor(tokenized_input['input_ids']).reshape(1,-1).to(device)
    token_type_ids = torch.tensor(tokenized_input['token_type_ids']).reshape(1,-1).to(device)
    word_ids = tokenized_input.word_ids() 

    logits = model(input_ids, attention_mask=attention_mask, token_type_ids = token_type_ids).logits[0].cpu()
    classes = logits[:,0:2]
    scores = torch.nn.functional.sigmoid(logits[:,2])

    phrases = []
    importance_scores = []
    i = 0
    while(i<len(scores)):
        if word_ids[i] == None or token_type_ids[0][i] == 0:
            i += 1 
            continue
        cl = torch.argmax(classes[i,:])
        if word_ids[i] == 0 or cl == 0: #we handle the edge case as well (beginning of the sentence)
            for j in range(i+1, len(scores)):
                cl = torch.argmax(classes[j,:])
                continue_word = False
                for k in range(i,j):
                    if word_ids[k] == word_ids[j]:
                        continue_word = True
                if (cl == 0 or  word_ids[j] == None) and continue_word == False:
                    break
            phrases.append(tokenizer.decode(input_ids[0][i:j]))
            importance_scores.append(scores[i].item())
            i = j 


    #maybe modify phrase with actual sentence
    real_phrases = []
    phrase_ind  = 0
    i = 0
    answer = answer.strip()

    while(i < len(answer)):
        last_token_place  = -1
        for j in range(i+1, len(answer)+1):
            if  phrases[phrase_ind].strip().replace(" ", "") == tokenizer.decode(tokenizer.encode(answer[i:j])[1:-1]).strip().replace(" ", ""):
                last_token_place = j

        real_phrases.append(answer[i:last_token_place ].strip())
        i = last_token_place
        phrase_ind += 1
        
    return real_phrases, importance_scores


def get_importance_vector(cleaned_sequence):
    importance_vector = []
    answer_ids = cleaned_sequence['cleaned_most_likely_generation_ids'][len(cleaned_sequence['prompt']):]
    #answer_ids = answer_ids[0:100]#normally it shouldn't be longer than 256
    answer = tokenizer.decode(answer_ids)
    question = cleaned_sequence['question']
    phrases,importance_vector = inference(model_importance, tokenizer_importance,question, answer)
    
    return torch.tensor(importance_vector), phrases



result_dict = {}
deberta_predictions = []
total_sample = len(sequences)
multiple_answer = 0
clusterable_semantic = 0 
for sample in tqdm(sequences):
    question = sample['question']
    generated_texts = sample['cleaned_generated_texts']

    id_ = sample['id'][0]

    unique_generated_texts = list(set(generated_texts))

    answer_list_1 = []
    answer_list_2 = []
    has_semantically_different_answers = False
    inputs = []
    syntactic_similarities = {}

    semantic_set_ids = {}
    for index, answer in enumerate(unique_generated_texts):
        semantic_set_ids[answer] = index


    importance_vector = get_importance_vector(sample)
    importance_scores = []
    
    generations = sample['cleaned_generations'].to(device)
    prompt = sample['prompt']
    
    for generation_index in range(generations.shape[0]):
        sequence = {}
        
        prompt = prompt[prompt != 1]
        generation = generations[generation_index][generations[generation_index] != 1]

        sequence['cleaned_most_likely_generation_ids'] = generation
        sequence['prompt'] = prompt
        sequence['question'] = sample['question']

        importance_scores.append(get_importance_vector(sequence))


    #print('Number of unique answers:', len(unique_generated_texts))#most of the time, it is 1 (interesting)
    has_different_answers = False
    encoded_meanings = []
    encoded_meanings_only_answer = []
    unique_answers_indices = []
    
    if len(unique_generated_texts) > 1:
        has_different_answers = True
        multiple_answer += 1 
        # Evalauate semantic similarity
        clusterable = False


        for i, reference_answer in enumerate(unique_generated_texts):
            q_a = question + ' ' + unique_generated_texts[i]
            a = unique_generated_texts[i]

            unique_answers_indices.append(generated_texts.index(a))
            

        for i, reference_answer in enumerate(unique_generated_texts):
            for j in range(i + 1, len(unique_generated_texts)):

                answer_list_1.append(unique_generated_texts[i])
                answer_list_2.append(unique_generated_texts[j])
        
                qa_1 = question + ' ' + unique_generated_texts[i]
                qa_2 = question + ' ' + unique_generated_texts[j]

                input = qa_1 + ' [SEP] ' + qa_2
                inputs.append(input)
                encoded_input = semantic_tokenizer.encode(input, padding=True)
                prediction = semantic_model(torch.tensor(torch.tensor([encoded_input]), device=device))['logits']
                predicted_label = torch.argmax(prediction, dim=1)

                reverse_input = qa_2 + ' [SEP] ' + qa_1
                encoded_reverse_input = semantic_tokenizer.encode(reverse_input, padding=True)
                reverse_prediction = semantic_model(torch.tensor(torch.tensor([encoded_reverse_input]), device=device))['logits']
                reverse_predicted_label = torch.argmax(reverse_prediction, dim=1)

                deberta_prediction = 1
                #print(qa_1, qa_2, predicted_label.item(), reverse_predicted_label.item())
                if 0 in predicted_label or 0 in reverse_predicted_label:
                    has_semantically_different_answers = True
                    deberta_prediction = 0
                else:
                    semantic_set_ids[unique_generated_texts[j]] = semantic_set_ids[unique_generated_texts[i]] # set same index to semantically similar sentences
                    clusterable = True

                deberta_predictions.append([unique_generated_texts[i], unique_generated_texts[j], deberta_prediction])

        if clusterable == True: 
            clusterable_semantic += 1
                
     

        # Evalauate syntactic similarity
        answer_list_1 = []
        answer_list_2 = []
        for i in generated_texts:
            for j in generated_texts:
                if i != j:
                    answer_list_1.append(i)
                    answer_list_2.append(j)

        

    
    result_dict[id_] = {
        'syntactic_similarities': syntactic_similarities,
        'has_semantically_different_answers': has_semantically_different_answers,
        'has_different_answers': has_different_answers
    }

    result_dict[id_]['importance_vector'] = importance_vector
    result_dict[id_]['importance_scores'] = importance_scores


    list_of_semantic_set_ids = [semantic_set_ids[x] for x in generated_texts]
    result_dict[id_]['semantic_set_ids'] = list_of_semantic_set_ids
    result_dict[id_]['unique_answers_indices'] = unique_answers_indices


with open(f'{config.output_dir}/sequences/{run_name}/{model_name}_generations_similarities.pkl', 'wb') as outfile:
    pickle.dump(result_dict, outfile)
