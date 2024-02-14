import argparse
import os
import pathlib
import pickle
from lib2to3.pgen2.tokenize import tokenize

import accelerate
#import config
import datasets
import evaluate
import numpy as np
import torch
from tqdm import tqdm
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import StoppingCriteria
import openai
from scipy.special import softmax
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import time
from flair.data import Sentence
from flair.models import SequenceTagger
import re
import json
import os
import random

seed_value = 10
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)

parser = argparse.ArgumentParser()
parser.add_argument('--fraction_of_data_to_use', type=float, default=0.9)
parser.add_argument('--run_id', type=str, default='run_1')
parser.add_argument('--temperature', type=float, default='0.5')
parser.add_argument('--num_beams', type=int, default='1')
parser.add_argument('--decoding_method', type=str, default='beam_search')
parser.add_argument('--top_p', type=float, default=1.0)
parser.add_argument('--dataset', type=str, default='trivia_qa')
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--model_name', type=str, default='mistralai/Mistral-7B-v0.1')
args = parser.parse_args()


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only use the specified GPUs
    try:
        tf.config.experimental.set_visible_devices(gpus[args.device:args.device+1], 'GPU')
    except RuntimeError as e:
        print(e)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Set the GPU memory limit or enable memory growth
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.experimental.set_virtual_device_configuration(
                gpu,
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*3)])  # For 3GB per GPU
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


VOCAB_PATH = 'gs://cloud-tpu-checkpoints/bert/keras_bert/uncased_L-12_H-768_A-12/vocab.txt'  #@param {type:"string"}
device = torch.device("cuda:" + str(args.device) if torch.cuda.is_available() else "cpu")

vocab_table = tf.lookup.StaticVocabularyTable(
        tf.lookup.TextFileInitializer(
            filename=VOCAB_PATH,
            key_dtype=tf.string,
            key_index=tf.lookup.TextFileIndex.WHOLE_LINE,
            value_dtype=tf.int64,
            value_index=tf.lookup.TextFileIndex.LINE_NUMBER
        ), 
        num_oov_buckets=1)
cls_id, sep_id = vocab_table.lookup(tf.convert_to_tensor(['[CLS]', '[SEP]']))
bert_tokenizer = text.BertTokenizer(vocab_lookup_table=vocab_table, 
                            token_out_type=tf.int64, 
                            preserve_unused_token=True, 
                            lower_case=True)

def bertify_example(example):
  question = bert_tokenizer.tokenize(example['question']).merge_dims(1, 2)
  reference = bert_tokenizer.tokenize(example['reference']).merge_dims(1, 2)
  candidate = bert_tokenizer.tokenize(example['candidate']).merge_dims(1, 2)

  input_ids, segment_ids = text.combine_segments(
      (candidate, reference, question), cls_id, sep_id)

  return {'input_ids': input_ids.numpy(), 'segment_ids': segment_ids.numpy()}


def pad(a, length=512):
  return np.append(a, np.zeros(length - a.shape[-1], np.int32))


def bertify_examples(examples):
  input_ids = []
  segment_ids = []
  for example in examples:
    example_inputs = bertify_example(example)
    input_ids.append(pad(example_inputs['input_ids']))
    segment_ids.append(pad(example_inputs['segment_ids']))

  return {'input_ids': np.stack(input_ids), 'segment_ids': np.stack(segment_ids)}


bem = hub.load('https://tfhub.dev/google/answer_equivalence/bem/1')
model = AutoModelForCausalLM.from_pretrained(args.model_name,
                                             torch_dtype=torch.float16).to(device)
tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)

if args.dataset == 'coqa':
    dataset = datasets.load_from_disk(f'../coqa_dataset')
    id_to_question_mapping = dict(zip(dataset['id'], dataset['question']))
elif args.dataset == 'trivia_qa':
    print('Preprocessing dataset')
    val_data = datasets.load_dataset("trivia_qa", "rc.nocontext", split="validation")
    train_data = datasets.load_dataset("trivia_qa", "rc.nocontext", split="train")
    data_for_few_shot_prompt = train_data.select(range(0, 10))

    few_shot_prompt = f"""Answer these questions:
            Question: What is the capital city of Australia?
            Answer: The capital city of Australia is Canberra.
            Question: Who painted the famous artwork "Starry Night"?
            Answer: "Starry Night" was painted by Vincent van Gogh.
            """

    #Question: Which planet is known as the "Red Planet"?
    #Answer: Mars
    batch_size = 4  # change to 16 for full training
    encoder_max_length = 1024
    decoder_max_length = 128

    def process_data_to_model_inputs(batch):
        # tokenize the inputs and labels
        answers = [answer["value"] for answer in batch["answer"]]


        #batch_with_prompt = [few_shot_prompt + "Question: " + question + " Answer:" for question in batch["question"]]
        batch_with_prompt = [f"""Answer these questions:
Question: What is the capital city of Australia?
Answer: The capital city of Australia is Canberra.
Question: Who painted the famous artwork "Starry Night"?
Answer: "Starry Night" was painted by Vincent van Gogh.
Question: {question} 
Answer:""" for question in batch["question"]]

        inputs = tokenizer(batch_with_prompt, padding=False, truncation=False)
        outputs = tokenizer(answers, padding=False, truncation=False)

        batch["input_ids"] = inputs.input_ids
        batch["attention_mask"] = inputs.attention_mask
        batch["decoder_input_ids"] = outputs.input_ids
        batch["decoder_attention_mask"] = outputs.attention_mask
        batch['answer'] = answers
        batch["labels"] = outputs.input_ids.copy()

        #Ithink below is not used at all

        # because BERT automatically shifts the labels, the labels correspond exactly to `decoder_input_ids`.
        # We have to make sure that the PAD token is ignored
        batch["labels"] = [
            [-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]
        ]

        return batch

    train_data = train_data.map(process_data_to_model_inputs,
                            batched=True,
                            batch_size=batch_size,
                            remove_columns=["search_results", "question_source", "entity_pages"])
    train_data.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
        output_all_columns=True)

    dataset = train_data

elif args.dataset == 'natural_qa':
    def get_fs_samples_prompt():
        data = datasets.load_dataset("nq_open", split='train')
        indices = np.random.RandomState(42).choice(len(data), 5)
        ret = ''
        for i in indices:
            i = int(i)
            ret += '\nQ: ' + data[i]['question'] + '\nA: ' + data[i]['answer'][0]
        return ret
    def sample_to_prompt(sample, **kwargs):
        return f"""Answer these questions:
Question: What is the capital city of Australia?
Answer: The capital city of Australia is Canberra.
Question: Who painted the famous artwork "Starry Night"?
Answer: "Starry Night" was painted by Vincent van Gogh.
Question: {sample['question']}?
Answer:"""

    def get_dataset(tokenizer):
        # For Natural Questions we use the test split used for open-domain question answering containing 3610 questions.
        data = datasets.load_dataset("nq_open", split='train')
        id_map = {_['question']:str(i) for i, _ in enumerate(data)}

        def process_instance(example):
            example['question_id'] = id_map[example['question']]
            all_answers = example.pop('answer')
            example['additional_answers'] = all_answers[1:]
            example['answer'] = all_answers[0]
            example['prompt'] = sample_to_prompt({k:example[k] for k in ['question']})
            inputs = tokenizer(example['prompt'], padding=False, truncation=False)
            outputs = tokenizer(all_answers[0], padding=False, truncation=False)
            example['input_ids'] = inputs['input_ids']
            example["attention_mask"] = inputs.attention_mask
            example["labels"] = outputs.input_ids.copy()
            example["labels"] = [-100 if _ == tokenizer.pad_token_id else _ for _ in example["labels"]]
            return example
        data = data.map(process_instance, load_from_cache_file=False)
        data.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "labels"],
            output_all_columns=True)
        return data
    
    dataset = get_dataset(tokenizer)


if args.fraction_of_data_to_use < 1.0:
    train_dataset = dataset.train_test_split(test_size=(1 - args.fraction_of_data_to_use), seed=seed_value)['train']
else:
    train_dataset = dataset

def encode(examples):
    return tokenizer(examples['story'] + ' Q: ' + examples['question'] + ' A:', truncation=False, padding=False)

def encode_and_format_dataset(dataset):
    dataset = dataset.map(encode, batched=False, load_from_cache_file=False)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'], output_all_columns=True)
    return dataset

if args.dataset == 'coqa':
    questions = encode_and_format_dataset(train_dataset)
elif args.dataset == 'trivia_qa':
    questions = train_dataset
elif args.dataset == 'natural_qa':
    questions = train_dataset


dataloader = torch.utils.data.DataLoader(questions, batch_size=1)


if tokenizer.__class__.__name__ == 'LlamaTokenizer':
    #eos_token_id = [tokenizer.encode(_)[-1] for _ in ['.']] + [29889]  # seems to be '.' as well
    eos_token_id = [tokenizer.encode(_)[-1] for _ in ['.', '\n']] + [29889]  # seems to be '.' as well
    if 'mistral' in args.model_name:
        eos_token_id += [28723]
        print('added additional eos token')
    #eos_token_id = [tokenizer(_)['input_ids'] for _ in ['\n', ',', '.']]
elif tokenizer.__class__.__name__ == 'GPT2Tokenizer':
    eos_token_id = [tokenizer.encode(_)[1] for _ in ['.', '\n']]
elif tokenizer.__class__.__name__ == 'PreTrainedTokenizerFast':
    eos_token_id = [tokenizer.encode(_)[-1] for _ in ['.', '\n']]
    eos_token_id += [691]
else:
    raise NotImplementedError

eos_token_id += [tokenizer.eos_token_id]

print(eos_token_id)

period_token_id = tokenizer('. ')['input_ids'][1]
eos_tokens = ['Question:', ' Question:', '\n', 'Answer:', ' Answer:', 'Q:']
question_framing_ids = [[tokenizer(eos_token)['input_ids'][-1]] for eos_token in eos_tokens]
squad_metric = evaluate.load("squad")
rouge = evaluate.load('rouge')
exact_match_metric = evaluate.load("exact_match")


tagger = SequenceTagger.load("flair/chunk-english")

def check_exist(words, tagged_sentence):
    for i, chunks in enumerate(tagged_sentence):
        if words.replace(" ", "").lower() == chunks.text.replace(" ", "").lower():
            return True, i
    return False, -1

def neural_phrase_tokenizer(sentence):
    tokenized_sentence = []
    tagged_sentence = Sentence(sentence)
    tagger.predict(tagged_sentence)

    words = re.findall(r'\w+|[^\w\s]', sentence)

    i = 0
    while(i < len(words)):
        found = False
        for j in range(i+1, len(words)):
            combined_word = " ".join(words[i:j])
            exist, index = check_exist(combined_word, tagged_sentence.get_spans('np'))
            if exist:
                tokenized_sentence.append(tagged_sentence.get_spans('np')[index].text)
                found = True
                i = j
        if found == False:
            tokenized_sentence.append(words[i])
            i = i + 1
    return tokenized_sentence

def get_importance_vector_BEM_phrase(answer_text, question_text):
    importance_vector = []

    print(answer_text)
    #words = answer.split()
    phrases = neural_phrase_tokenizer(answer_text)
    print(phrases)
    
    #encoded_answer = sentence_model.encode(answer)

    for i in range(len(phrases)):
        removed_answer = phrases[:i] +  phrases[i+1:]

        removed_answer = ' '.join(removed_answer )

        print(removed_answer)
        bem_input = [{
            'question': question_text,
            'reference': answer_text,
            'candidate': removed_answer
            }]

        inputs = bertify_examples(bem_input)
        raw_outputs = bem(inputs)
        bem_score = float(softmax(np.squeeze(raw_outputs))[1])
        print(1-bem_score)
        score = 1 - bem_score
        importance_vector.append(score)

    importance_vector = np.array(importance_vector)

    return importance_vector, phrases


def label_phrases(phrases, importance_scores):
    words = []
    scores = []
    labels = []
    for i, phrase in enumerate(phrases):
        phrase_words = re.findall(r'\w+|[^\w\s]', phrase)
        for j,phrase_word in enumerate(phrase_words):
            words.append(phrase_word)
            if j == 0:
                scores.append(importance_scores[i])
                labels.append("B-CNK")
            else:
                scores.append(-100)
                labels.append("I-CNK")
    return words,scores,labels


def get_generations(model, dataloader):
    with torch.no_grad():
        max_length_of_generated_sequence = 128
        importance_samples = []
        for batch in tqdm(dataloader):
            importance_sample = {}
            input_ids = batch['input_ids'].to(device).reshape(
                1, -1) if args.dataset == 'trivia_qa' or args.dataset == 'natural_qa'  else batch['input_ids'].to(device)

            most_likely_generation = model.generate(input_ids,
                                                    num_beams=1,
                                                    num_return_sequences=1,
                                                    do_sample=False,
                                                    max_length=input_ids.shape[1] +
                                                    max_length_of_generated_sequence,
                                                    #eos_token_id=period_token_id,
                                                    eos_token_id = eos_token_id,
                                                    pad_token_id =tokenizer.eos_token_id,
                                                    bad_words_ids=question_framing_ids)


            most_likely_generation_text =  tokenizer.decode(
                    most_likely_generation[0][len(batch['input_ids'][0]):], skip_special_tokens=True)

            few_shot_question = tokenizer.decode(input_ids[0])
            question_text = few_shot_question.split('Question: ')[-1].split('\nAnswer:')[0]

            #create labels
            importance_scores, phrases  = get_importance_vector_BEM_phrase(most_likely_generation_text, question_text)

            #post procees phrases
            words, scores, labels = label_phrases(phrases, importance_scores)

            importance_sample['question'] = question_text
            importance_sample['answer_text'] =  most_likely_generation_text
            importance_sample['words'] =  words                         
            importance_sample['scores'] =  scores
            importance_sample['labels'] =  labels  
            importance_sample['phrases'] =  phrases

            importance_samples.append(importance_sample)
        
    return importance_samples

importance_samples = get_generations(model, dataloader)
data_name = f"./data/{args.dataset}_{args.model_name.replace('/', '_')}.json"
with open(data_name, "w") as outfile:
    outfile.write(json.dumps(importance_samples))
