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
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import StoppingCriteria
import openai
from scipy.special import softmax
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import time
from datasets import load_dataset




openai.api_key = "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"#placeholder

parser = argparse.ArgumentParser()
parser.add_argument('--type_of_question', type=str)
parser.add_argument('--num_generations_per_prompt', type=int, default=5)
parser.add_argument('--fraction_of_data_to_use', type=float, default=0.9)
parser.add_argument('--run_id', type=str, default='run_1')
parser.add_argument('--temperature', type=float, default='1.0')
parser.add_argument('--num_beams', type=int, default='1')
parser.add_argument('--decoding_method', type=str, default='beam_search')
parser.add_argument('--top_p', type=float, default=1.0)
parser.add_argument('--dataset', type=str, default='trivia_qa')
parser.add_argument('--device', type=int, default=0)

parser.add_argument('--model_name', type=str, default='mistralai/Mistral-7B-v0.1')

args = parser.parse_args()


# Set a seed value
seed_value = 10
# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED'] = str(seed_value)
# 2. Set `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)
# 3. Set `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)
#Fix torch random seed
torch.manual_seed(seed_value)

VOCAB_PATH = 'gs://cloud-tpu-checkpoints/bert/keras_bert/uncased_L-12_H-768_A-12/vocab.txt'  #@param {type:"string"}

tf.config.set_visible_devices([], 'GPU')

device = torch.device("cuda:" + str(args.device) if torch.cuda.is_available() else "cpu")

# Use the specified GPU device

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

run_name = args.run_id


if args.dataset == 'trivia_qa':
    print('Preprocessing dataset')
    val_data = datasets.load_dataset("trivia_qa", "rc.nocontext", split="validation")
    train_data = datasets.load_dataset("trivia_qa", "rc.nocontext", split="train")
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
Answer: """ for question in batch["question"]]

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

    val_data = val_data.map(process_data_to_model_inputs,
                            batched=True,
                            batch_size=batch_size,
                            remove_columns=["search_results", "question_source", "entity_pages"])
    val_data.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
        output_all_columns=True)

    dataset = val_data
elif args.dataset == 'natural_qa':
    def sample_to_prompt(sample, **kwargs):
        return f"""Answer these questions:
Question: What is the capital city of Australia?
Answer: The capital city of Australia is Canberra.
Question: Who painted the famous artwork "Starry Night"?
Answer: "Starry Night" was painted by Vincent van Gogh.
Question: {sample['question']}?
Answer: """

    def get_dataset(tokenizer):
        # For Natural Questions we use the test split used for open-domain question answering containing 3610 questions.
        data = datasets.load_dataset("nq_open", split='validation')
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

elif args.dataset == 'web_qa':
    print('Preprocessing dataset')
    test_data = datasets.load_dataset("web_questions", "default", split="test")
    train_data = datasets.load_dataset("web_questions", "default", split="train")
    val_data = datasets.concatenate_datasets([train_data, test_data])
    batch_size = 4  # change to 16 for full training
    encoder_max_length = 1024
    decoder_max_length = 128

    def process_data_to_model_inputs(batch):
        #print(batch)
        # tokenize the inputs and labels
        all_answers = [answer[0] for answer in batch["answers"]]
        all_additional_answers = [answer[1:] for answer in batch["answers"]]

        #batch_with_prompt = [few_shot_prompt + "Question: " + question + " Answer:" for question in batch["question"]]
        batch_with_prompt = [f"""answer these questions:
Question: what is the capital city of australia?
Answer: the capital city of australia is canberra.
Question: who painted the famous artwork "starry night"?
Answer: "starry night" was painted by vincent van gogh.
Question: {question} 
Answer: """ for question in batch["question"]]

        inputs = tokenizer(batch_with_prompt, padding=False, truncation=False)

        batch["input_ids"] = inputs.input_ids
        batch["attention_mask"] = inputs.attention_mask

        batch['additional_answers'] = all_additional_answers
        batch['answer'] = all_answers
        

        outputs = tokenizer(all_answers, padding=False, truncation=False)
        batch["decoder_input_ids"] = outputs.input_ids
        batch["decoder_attention_mask"] = outputs.attention_mask
        batch["labels"] = outputs.input_ids.copy()

        batch['question_id'] =  [question for question in batch["question"]] #unique id for each question, assume each question is unique

        #Ithink below is not used at all

        # because BERT automatically shifts the labels, the labels correspond exactly to `decoder_input_ids`.
        # We have to make sure that the PAD token is ignored
        batch["labels"] = [
            [-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]
        ]

        return batch

    val_data = val_data.map(process_data_to_model_inputs,
                            batched=True,
                            batch_size=batch_size,
                            remove_columns=["url"])



    val_data.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
        output_all_columns=True)

    dataset = val_data
elif args.dataset == 'medical_dataset':
    print('Preprocessing dataset')
    train_data = load_dataset("json", data_files="medical_dataset.json")['train']
    batch_size = 4  # change to 16 for full training
    encoder_max_length = 1024
    decoder_max_length = 128

    def process_data_to_model_inputs(batch):
        #print(batch)
        # tokenize the inputs and labels
        all_answers = [answer[0] for answer in batch["answer"]]
        all_additional_answers = [answer[1:] for answer in batch["answer"]]
        #batch_with_prompt = [few_shot_prompt + "Question: " + question + " Answer:" for question in batch["question"]]
        batch_with_prompt = [f"""You are a medical doctor answering real-world medical entrance exam questions. Based on your understanding of basic and clinical science, medical knowledge, and mechanisms underlying health, disease, patient care, and modes of therapy, answer the following closed-end questions. Give one sentence straight response. Question: What is the primary organ affected by hepatitis? Answer: The liver is the primary organ affected by hepatitis. Question: What is the significance of the CRISPR-Cas9 system in genetic engineering? Answer: CRISPR-Cas9 allows for precise editing of the DNA in organisms, offering potential treatments for genetic disorders by correcting gene mutations. Question: What is the primary cause of cystic fibrosis? Answer:Cystic fibrosis is primarily caused by mutations in the CFTR gene affecting chloride ion transport in cells. Question: How does HIV evade the immune system? Answer: HIV evades the immune system by rapidly mutating and hiding within T-cells. Question: How do statins lower cholesterol? Answer: Statins work by inhibiting HMG-CoA reductase, an enzyme involved in cholesterol production in the liver. Question: {question.strip()}? Answer: """ for question in batch["question"]]      
        inputs = tokenizer(batch_with_prompt, padding=False, truncation=False)
        batch["input_ids"] = inputs.input_ids
        batch["attention_mask"] = inputs.attention_mask

        batch['additional_answers'] = all_additional_answers
        batch['answer'] = all_answers
        

        outputs = tokenizer(all_answers, padding=False, truncation=False)
        batch["decoder_input_ids"] = outputs.input_ids
        batch["decoder_attention_mask"] = outputs.attention_mask
        batch["labels"] = outputs.input_ids.copy()

        batch['question_id'] =  [question for question in batch["question"]] #unique id for each question, assume each question is unique

        #Ithink below is not used at all

        # because BERT automatically shifts the labels, the labels correspond exactly to `decoder_input_ids`.
        # We have to make sure that the PAD token is ignored
        batch["labels"] = [
            [-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]
        ]

        return batch

    train_data = train_data.map(process_data_to_model_inputs,
                            batched=True,
                            batch_size=batch_size)



    train_data.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
        output_all_columns=True)

    dataset = train_data




if args.fraction_of_data_to_use < 1.0:
    train_dataset = dataset.train_test_split(test_size=(1 - args.fraction_of_data_to_use), seed=seed_value)['train']
else:
    train_dataset = dataset

questions = train_dataset
dataloader = torch.utils.data.DataLoader(questions, batch_size=1)

#Setting `pad_token_id` to `eos_token_id`:842 for open-end generation.

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
elif tokenizer.__class__.__name__ == 'CodeGenTokenizer':
    eos_token_id = [tokenizer.encode(_)[-1] for _ in ['.']]
    #eos_token_id += [691]
else:
    raise NotImplementedError

eos_token_id += [tokenizer.eos_token_id]

period_token_id = tokenizer('. ')['input_ids'][1]
if args.dataset == 'medical_dataset':
    print('medical dataset')
    eos_tokens = ['Question:', ' Question:', '\n', 'Answer:', ' Answer:', 'Q:',"/fig", "/table","[/fig]","[/table]","<EOD>","1)","1.",]
    question_framing_ids = [tokenizer.encode(eos_token, add_special_tokens=False) for eos_token in eos_tokens]
    print('question framing ids')
else:
    eos_tokens = ['Question:', ' Question:', '\n', 'Answer:', ' Answer:', 'Q:']
    question_framing_ids = [[tokenizer(eos_token)['input_ids'][-1]] for eos_token in eos_tokens]

exact_match_metric = evaluate.load("exact_match")
max_retries = 30  # Maximum number of retry attempts
retry_delay = 2

def send_openai_request_with_retries(prompt):
    retries = 0

    while retries < max_retries:
        try:
            if args.dataset == 'medical_dataset':
                model_name = "gpt-4-turbo-preview"
            else:
                model_name ="gpt-3.5-turbo"

            completion = openai.ChatCompletion.create(
                model=model_name,
                seed=seed_value,
                messages=[{"role": "user", "content": f'{prompt}'}]
            )
            # If the request is successful, you can process the response here
            return completion
        except openai.error.APIError as e:
            print(f"OpenAI API request failed with error: {e}")
            retries += 1
            if retries < max_retries:
                print(f"Retrying OpenAI request (attempt {retries})...")
                time.sleep(retry_delay)
            else:
                print("Max retries reached. Unable to complete the OpenAI request.")
                return None 
        except openai.error.Timeout as e:
            print(f"OpenAI API request failed with error: {e}")
            retries += 1
            if retries < max_retries:
                print(f"Retrying OpenAI request (attempt {retries})...")
                time.sleep(retry_delay)
            else:
                print("Max retries reached. Unable to complete the OpenAI request.")
                return None
        except openai.error.ServiceUnavailableError as e:
            print(f"OpenAI API request failed with error: {e}")
            retries += 1
            if retries < max_retries:
                print(f"Retrying OpenAI request (attempt {retries})...")
                time.sleep(retry_delay)
            else:
                print("Max retries reached. Unable to complete the OpenAI request.")
                return None  
        except openai.error.RateLimitError as e:
            print(f"OpenAI API request failed with error: {e}")
            retries += 1
            if retries < max_retries:
                print(f"Retrying OpenAI request (attempt {retries})...")
                time.sleep(retry_delay)
            else:
                print("Max retries reached. Unable to complete the OpenAI request.")
                return None  
                
def get_generations(model, dataloader, number_of_generations):
    with torch.no_grad():
        max_length_of_generated_sequence = 128
        sequences = []
        for batch in tqdm(dataloader):
            input_ids = batch['input_ids'].to(device).reshape(1, -1) 
    
            #there is no sampling here. It will be used for evaluation
            if args.decoding_method == 'beam_search':
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
                                                    
            elif args.decoding_method == 'greedy':
                most_likely_generation = model.generate(input_ids,
                                                        num_beams=1,
                                                        do_sample=False,
                                                        max_length=input_ids.shape[1] +
                                                        max_length_of_generated_sequence,
                                                        #eos_token_id=period_token_id,
                                                        eos_token_id = eos_token_id,
                                                        bad_words_ids=question_framing_ids)


            #now generate random outputs to calculate uncertainty
            input_length = input_ids.shape[1]

            generations = torch.ones((args.num_generations_per_prompt, input_length + max_length_of_generated_sequence),
                                            dtype=torch.long,
                                            device=device)

            for i in range(args.num_generations_per_prompt):

                generation = model.generate(input_ids,
                                            do_sample=True,
                                            num_return_sequences=1,
                                            num_beams=args.num_beams,
                                            max_length=input_ids.shape[1] + max_length_of_generated_sequence,
                                            #eos_token_id=period_token_id,
                                            eos_token_id =eos_token_id,
                                            temperature=args.temperature,
                                            pad_token_id =tokenizer.eos_token_id,
                                            bad_words_ids=question_framing_ids,
                                            top_p=args.top_p)
                generations[i, :generation.shape[1]] = generation

            generations = torch.reshape(generations, (-1, args.num_generations_per_prompt, generations.shape[-1]))

            for i in range(generations.shape[0]):
                few_shot_question = tokenizer.decode(input_ids[0])
                question = few_shot_question.split('Question: ')[-1].split('Answer: ')[0]
                #there are 2 examples having same id:
                random_number = np.random.rand(1)[0]
                #print(random_number)
                batch['question_id'][0] = batch['question_id'][0]+str(random_number)
                sequence_dict = {
                    'prompt': input_ids[0],
                    'generations': generations[i],
                    'id': batch['question_id'],
                    'few_shot_question': tokenizer.decode(input_ids[0]),
                    'question': question
                }

                generated_texts = []
                for generation in generations[i]:
                    generated_texts.append(
                        tokenizer.decode(generation[len(batch['input_ids'][i]):], skip_special_tokens=True))#We already skip special tokens

                sequence_dict['generated_texts'] = generated_texts
                sequence_dict['most_likely_generation_ids'] = most_likely_generation[0].to('cpu')
                sequence_dict['most_likely_generation'] = tokenizer.decode(
                    most_likely_generation[0][len(batch['input_ids'][i]):], skip_special_tokens=True)

                sequence_dict['second_most_likely_generation_ids'] = most_likely_generation[0].to('cpu')#These are not used at all
                sequence_dict['second_most_likely_generation'] = tokenizer.decode(
                    most_likely_generation[0][len(batch['input_ids'][i]):], skip_special_tokens=True)

                #There are different answers for each question. These rauge scores show how they are related to each other.
                sequence_dict['answer'] = batch['answer']
                
                sequence_dict['exact_match'] = 0.0

                sequence_dict['bem_score'] = 0.0

                sequence_dict['gpt_score'] = 0.0

                if args.dataset=='natural_qa':
                    reference_answers = batch['answer'] + [x[0] for x in batch['additional_answers']]
                elif args.dataset=='web_qa':
                    reference_answers = batch['answer'] + [x[0] for x in batch['additional_answers']]
                elif args.dataset=='medical_dataset':
                    reference_answers = batch['answer'] + [x[0] for x in batch['additional_answers']]
                else:
                    reference_answers = batch['answer']
                
                #print(batch['answer'])
                sequence_dict['all_answers'] = reference_answers
                
                for answer in reference_answers:
                    predictions = [sequence_dict['most_likely_generation'].lstrip()]
                    references = [answer]
                    results = exact_match_metric.compute(predictions=predictions,
                                                            references=references,
                                                            ignore_case=True,
                                                            ignore_punctuation=True)
                    sequence_dict['exact_match'] = max(results['exact_match'], sequence_dict['exact_match'])
                    examples = [{
                    'question': batch['question'][0],
                    'reference': answer,
                    'candidate': sequence_dict['most_likely_generation'].lstrip()
                    }]

                
                    inputs = bertify_examples(examples)

                    # The outputs are raw logits.
                    raw_outputs = bem(inputs)
                    # They can be transformed into a classification 'probability' like so:
                    bem_score = float(softmax(np.squeeze(raw_outputs))[1])
                    sequence_dict['bem_score'] = max(bem_score,sequence_dict['bem_score'])

                    
                    #context = tokenizer.decode(sequence_dict['prompt'], skip_special_tokens=True)
                    #print(context[:-3])
                    if args.dataset=='medical_dataset':
                        prompt = f'''You will behave as a medical question answer evaluator. I will give you a medical question, generated answer by a language model. \
Use your own medical knowledge. You will output "correct" if the generated answer is correct regarding question and your medical knowledge. Otherwise, output "false".
Question: {batch["question"][0]}?,
Generated Answer: {sequence_dict["most_likely_generation"].lstrip()}'''
                    else:
                        prompt = f'''You will behave as an question answer evaluator. I will give you a question, the ground truth of the question and a generated answer by a language model. You will output "correct" if the generated answer is correct regarding question and ground truth. \
Otherwise, output "false".
Question: {batch["question"][0]}, 
Ground Truth: {answer},
Generated Answer: {sequence_dict["most_likely_generation"].lstrip()}'''

                    completion =  send_openai_request_with_retries(prompt)
                    if completion.choices[0].message.content.lower() == 'correct' or completion.choices[0].message.content.lower() == 'correct.':
                        gpt_score = 100
                    elif completion.choices[0].message.content.lower() == 'false' or completion.choices[0].message.content.lower() == 'false.':
                        gpt_score = 0
                    else:
                        print("BUGGY ANSWER")
                        gpt_score = bem_score * 100
                    
                    sequence_dict['gpt_score'] = max(gpt_score,sequence_dict['gpt_score'])
                    

                sequences.append(sequence_dict)
                
    return sequences


sequences = get_generations(model, dataloader, args.num_generations_per_prompt)

pathlib.Path(f'{config.output_dir}/sequences/' + run_name).mkdir(parents=True, exist_ok=True)

model_name = args.model_name.replace("/", "")
with open(f'{config.output_dir}/sequences/{run_name}/{model_name}_generations.pkl', 'wb') as outfile:
    pickle.dump(sequences, outfile)