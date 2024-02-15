import json
import transformers
from datasets import load_dataset
import datasets 
import torch
import numpy as np 
from transformers import BertTokenizerFast 
from transformers import DataCollatorForTokenClassification 
from transformers import AutoModelForTokenClassification 
from scipy.special import softmax
import re
import time
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--temperature', type=float, default=0.01)
parser.add_argument('--lamda', type=float, default=0.5)
parser.add_argument('--lr', type=float, default=5e-5)
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--device', type=int, default=0)

args = parser.parse_args()

# file_paths = ['data/natural_qa_meta-llama_Llama-2-7b-chat-hf.json',
#   'data/natural_qa_mistralai_Mistral-7B-v0.1.json',  'data/trivia_qa_meta-llama_Llama-2-7b-chat-hf.json',  
#   'data/trivia_qa_mistralai_Mistral-7B-v0.1.json', 'data/natural_qa_meta-llama_Llama-2-7b-hf.json',
#     'data/natural_qa_tiiuae_falcon-7b-instruct.json',  'data/trivia_qa_meta-llama_Llama-2-7b-hf.json',
#     'data/trivia_qa_tiiuae_falcon-7b-instruct.json']

file_paths = ['data/natural_qa_meta-llama_Llama-2-7b-chat-hf_word.json',
  'data/natural_qa_mistralai_Mistral-7B-v0.1_word.json',  'data/trivia_qa_meta-llama_Llama-2-7b-chat-hf_word.json',  
  'data/trivia_qa_mistralai_Mistral-7B-v0.1_word.json', 'data/natural_qa_meta-llama_Llama-2-7b-hf_word.json',
    'data/natural_qa_tiiuae_falcon-7b-instruct_word.json',  'data/trivia_qa_meta-llama_Llama-2-7b-hf_word.json',
    'data/trivia_qa_tiiuae_falcon-7b-instruct_word.json']

dataset = load_dataset('json', data_files=file_paths, split="train")

test_data = dataset.train_test_split(test_size=(0.02), seed=0)['test']
training_data = dataset.train_test_split(test_size=(0.02), seed=0)['train']

def keep_sample(example):#sometimes model generates nothing, remove that sample
    if example['words'] == []:
        return False
    return True

test_data = test_data.filter(keep_sample)
training_data = training_data.filter(keep_sample)

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased") 
label_converter = {"B-CNK":0, "I-CNK":1}


def softmax_with_temperature(logits, temperature):
    #print(logits)
    scaled_logits = logits / temperature
    softmax_probs = softmax(scaled_logits, axis=0)
    return softmax_probs


def tokenize_and_align_labels(example, temperature=0.01): 
    """
    Function to tokenize and align labels with respect to the tokens. This function is specifically designed for
    Named Entity Recognition (NER) tasks where alignment of the labels is necessary after tokenization.

    Parameters:
    examples (dict): A dictionary containing the tokens and the corresponding NER tags.
                     - "tokens": list of words in a sentence.
                     - "ner_tags": list of corresponding entity tags for each word.
                     
    label_all_tokens (bool): A flag to indicate whether all tokens should have labels. 
                             If False, only the first token of a word will have a label, 
                             the other tokens (subwords) corresponding to the same word will be assigned -100.

    Returns:
    tokenized_inputs (dict): A dictionary containing the tokenized inputs and the corresponding labels aligned with the tokens.
    """


    tokenized_input = tokenizer.encode_plus(
        [example["question"]],
        example["words"],
        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        return_token_type_ids=True,
        is_split_into_words=True,
        truncation=True,
        max_length = 512,           # Pad & truncate all sentences.
        pad_to_max_length = True,
    )
    labels = np.ones(512) * -100 
    scores = np.ones(512) * -100
    word_ids = tokenized_input.word_ids() 
    token_type_ids = tokenized_input.token_type_ids
    label = example["labels"]

    previous_word_idx = None 
    # Special tokens like `<s>` and `<\s>` are originally mapped to None 
    # We need to set the label to -100 so they are automatically ignored in the loss function.
    for j, word_idx in enumerate(word_ids): 
        if word_idx is None or token_type_ids[j] == 0: 
            # set –100 as the label for these special tokens
            labels[j] = -100
            scores[j] = -100
        # For the other tokens in a word, we set the label to either the current label or -100, depending on
        # the label_all_tokens flag.
        elif word_idx != previous_word_idx:
            # if current word_idx is != prev then its the most regular case
            # and add the corresponding token                 
            labels[j] = (label_converter[label[word_idx]]) 
            scores[j] = (example["scores"][word_idx])
        else: 
            # to take care of sub-words which have the same word_idx
            # set -100 as well for them, but only if label_all_tokens == False
            #labels[j] = (label_converter[label[word_idx]]) 
            labels[j] = 1
            scores[j] = (-100)
            # mask the subword representations after the first subword
                
        previous_word_idx = word_idx 

    scores = np.array(scores)
    score_indices = np.where(scores != -100)
    scores[score_indices] = softmax_with_temperature(scores[score_indices], temperature)
    
    # tokenized_input["labels_token"] = labels 
    # tokenized_input["scores_token"] = scores 
    # return tokeinp

    return { "labels_token": torch.tensor(labels, dtype=int),
            "scores_token": torch.tensor(scores),
            'input_ids': torch.tensor(tokenized_input["input_ids"]), 
            'token_type_ids': torch.tensor(tokenized_input["token_type_ids"]), 
            'attention_mask': torch.tensor(tokenized_input["attention_mask"]),
    } 


def custom_collate_fn(batch):
    # Assuming each element in 'batch' is a dictionary with 'inputs' and 'labels'
    labels = [torch.tensor(item["labels_token"]) for item in batch]
    scores = [torch.tensor(item["scores_token"]) for item in batch]
    inps = [torch.tensor(item["input_ids"]) for item in batch]
    types = [torch.tensor(item["token_type_ids"]) for item in batch]
    masks = [torch.tensor(item["attention_mask"]) for item in batch]

    # Convert lists to tensors and stack them
    labels = torch.stack(labels)
    scores = torch.stack(scores)
    inps = torch.stack(inps)
    types = torch.stack(types)
    masks = torch.stack(masks)

    return labels, scores, inps, types, masks


tokenized_datasets = training_data.map(tokenize_and_align_labels, batched=False, fn_kwargs={"temperature": args.temperature})
test_data = test_data.map(tokenize_and_align_labels, batched=False, fn_kwargs={"temperature": args.temperature})
train_loader = torch.utils.data.DataLoader(tokenized_datasets, batch_size=32, shuffle=True, collate_fn=custom_collate_fn)

def test_loss(test_data, model, device):
    model.eval()    
    losses_class = []
    losses_score = []
    for i in range(len(test_data)):
        #test loss code
        labels_token = torch.tensor(test_data[i]['labels_token']).reshape(1,-1).to(device)
        scores_token = torch.tensor(test_data[i]['scores_token']).reshape(1,-1).to(device)
        input_ids = torch.tensor(test_data[i]['input_ids']).reshape(1,-1).to(device)
        attention_mask = torch.tensor(test_data[i]['attention_mask']).reshape(1,-1).to(device)
        token_type_ids = torch.tensor(test_data[i]['token_type_ids']).reshape(1,-1).to(device)

        logits = model(input_ids, attention_mask=attention_mask, token_type_ids = token_type_ids).logits.detach()
        loss_class, loss_score = importance_loss_with_clf(logits, labels_token,scores_token)
        losses_class.append(loss_class.item())
        losses_score.append(loss_score.item())

    losses_class = np.array(losses_class)
    losses_score = np.array(losses_score)

    return np.mean(losses_class), np.mean(losses_score)


def importance_loss_with_clf(logits, labels, scores):

    cross_loss = torch.nn.CrossEntropyLoss()
    bce_loss = torch.nn.BCEWithLogitsLoss()

    lab = labels[labels != -100]
    sco = scores[scores != -100].reshape(-1,1)
    log_sco = logits[scores != -100][:, 2:3]
    log_lab = logits[labels != -100][:, :2]

    #cross entropy loss
    loss1 = cross_loss(log_lab, lab)
    #bce loss
    loss2 = bce_loss(log_sco, sco)
    return loss1, loss2


def train(model, train_loader, test_data, lamda, epochs, lr, device):
    model.to(device)
    optimizer = transformers.AdamW(model.parameters(), lr=lr)
    iteration = 0
    train_loss1, train_loss2, total_sample = 0, 0, 0
    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        for (labels, scores, input_ids, token_type_ids, attention_mask) in tqdm(train_loader):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            labels = labels.to(device)
            scores = scores.to(device)

            # Forward pass
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask=attention_mask, token_type_ids = token_type_ids).logits
            loss1, loss2 = importance_loss_with_clf(logits, labels,scores)
            loss = loss1 * lamda + (1-lamda) * loss2
            loss.backward()
            optimizer.step()
            train_loss1 += loss1.item() * len(labels)
            train_loss2 += loss2.item() * len(labels)
            total_sample += len(labels)

            if (iteration) % 400 == 0:

                # Validation step
                model.eval()
                with torch.no_grad():
                    class_loss, score_loss = test_loss(test_data, model, device)
                
                end_time = time.time()
                print(f"iteration {iteration}  | Train cls loss: {train_loss1/total_sample:.2f}  | Train scr loss: {train_loss2/total_sample:.2f}\
        | Val cls loss: {class_loss:.2f}  | Val scr loss: {score_loss:.2f}  | {end_time-start_time:.2f}s")

                train_loss1, train_loss2, total_sample = 0, 0, 0
                model.train()
            iteration += 1
    return model



device = torch.device("cuda:" + str(args.device) if torch.cuda.is_available() else "cpu")
model = AutoModelForTokenClassification.from_pretrained("bert-base-uncased", num_labels=3).to(device)
model = train(model, train_loader, test_data, lamda=args.lamda, epochs=args.epochs, lr=args.lr, device=device)

model_name = f'importance_model_v1_t_{args.temperature}_word.pth'
torch.save(model, model_name )

