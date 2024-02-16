# MARS: Meaning Aware Response Scoring
This repository is built on the [Semantic Uncertainty Repository](https://github.com/lorenzkuhn/semantic_uncertainty)

## Installing Dependencies
To create the environment, run the following command:
```
conda env create -f mars.yml 
```
and run the following command with the path to your conda environment:
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path_to_conda/envs/mars/lib
```
then activate the environment:
```
conda activate mars
```
## Importance Model
You can download the importance model from the following [anonymous Google Drive link](https://drive.google.com/file/d/1HyhtNS2xtqJ6yKsdgnYv9lFUC-6Lcx2-/view?usp=share_link) and put model_phrase.pth inside models folder.

As a second option, you can train model yourself. First, you should create the labelled dataset using BEM model by running following commands:

```
python create_importance_dataset.py --model_name=mistralai/Mistral-7B-v0.1 --device=0 --dataset=trivia_qa --fraction_of_data_to_use='0.5'

python create_importance_dataset.py --model_name=tiiuae/falcon-7b-instruct --device=0 --dataset=trivia_qa --fraction_of_data_to_use='0.5'

python create_importance_dataset.py --model_name=meta-llama/Llama-2-7b-chat-hf --device=0 --dataset=trivia_qa --fraction_of_data_to_use='0.5'

python create_importance_dataset.py --model_name=meta-llama/Llama-2-7b-hf --device=0 --dataset=trivia_qa --fraction_of_data_to_use='0.5'


python create_importance_dataset.py --model_name=mistralai/Mistral-7B-v0.1 --device=0 --dataset=natural_qa --fraction_of_data_to_use='1.0'

python create_importance_dataset.py --model_name=tiiuae/falcon-7b-instruct --device=0 --dataset=natural_qa --fraction_of_data_to_use='1.0'

python create_importance_dataset.py --model_name=meta-llama/Llama-2-7b-chat-hf--device=0 --dataset=natural_qa --fraction_of_data_to_use='1.0'

python create_importance_dataset.py --model_name=meta-llama/Llama-2-7b-hf --device=0 --dataset=natural_qa --fraction_of_data_to_use='1.0'
```

then train a Bert-like with generated labelled data by running following command:
```
python fine_tune_bert_v1.py --device=0
```

## OpenAI Setup
To be able to run experiments, OpenAI key is required. Replace a valid OpenAI key in the following line in generate_answers.py:
```
openai.api_key = ""
```

## Running Experiments
To get the results of Table 1, run the bash file: run_framework.sh.
The arguments of run_framework.sh is following:

```
sh run_framework.sh $model_name $run_name  $gpu_no $dataset $fraction_of_data_to_use $temperature
```

Run following commands for TriviaQA:

```
sh run_framework.sh tiiuae/falcon-7b-instruct trivia_qa_tiiuae-falcon-7b-instruct 0 trivia_qa 0.5 0.5
sh run_framework.sh mistralai/Mistral-7B-v0.1 trivia_qa_mistralai-Mistral-7B 0 trivia_qa 0.5 0.5
sh run_framework.sh meta-llama/Llama-2-7b-chat-hf trivia_qa_meta-llama-Llama-2-7b-chat-hf 0 trivia_qa 0.5 0.5
sh run_framework.sh meta-llama/Llama-2-7b-hf trivia_qa_meta-llama-Llama-2-7b-hf 0 trivia_qa 0.5 0.5 
sh run_framework.sh meta-llama/Llama-2-13b-hf trivia_qa_meta-llama-Llama-2-13b-hf 0 trivia_qa 0.5 0.5
```

Run following commands for NaturalQA:
```
sh run_framework.sh tiiuae/falcon-7b-instruct natural_qa_tiiuae-falcon-7b-instruct 0 natural_qa 1.0 0.5
sh run_framework.sh mistralai/Mistral-7B-v0.1 natural_qa_mistralai-Mistral-7B-v0.1 0 natural_qa 1.0 0.5
sh run_framework.sh meta-llama/Llama-2-7b-chat-hf natural_qa_meta-llama-Llama-2-7b-chat-hf 0 natural_qa 1.0 0.5
sh run_framework.sh meta-llama/Llama-2-7b-hf natural_qa_meta-llama-Llama-2-7b-hf 0 natural_qa 1.0 0.5
sh run_framework.sh meta-llama/Llama-2-13b-hf natural_qa_meta-llama-Llama-2-13b-hf 0 natural_qa 1.0 0.5

```

Run following commands for WebQA:
```
#sh run_framework.sh  tiiuae/falcon-7b-instruct web_qa_tiiuae-falcon-7b-instruct 0 web_qa 1.0 0.5
#sh run_framework.sh  mistralai/Mistral-7B-v0.1 web_qa_mistralai-Mistral-7B-v0.1 0 web_qa 1.0 0.5
#sh run_framework.sh  meta-llama/Llama-2-7b-chat-hf web_qa_meta-llama-Llama-2-7b-chat-hf 0 web_qa 1.0 0.5
#sh run_framework.sh meta-llama/Llama-2-7b-hf web_qa_meta-llama-Llama-2-7b-hf 0 web_qa 1.0 0.5
#sh run_framework.sh meta-llama/Llama-2-13b-hf web_qa_meta-llama-Llama-2-13b-hf 0 web_qa 1.0 0.5
```

Run following commands for Medical QA:

```
sh run_framework.sh AdaptLLM/medicine-chat medical_dataset_qa 0 medical_dataset 1.0 0.25
```




