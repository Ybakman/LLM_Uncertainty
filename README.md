# MARS: Meaning Aware Response Scoring

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
## Downloading Datasets and Model

## Wandb and OpenAI Setup
To be able to run experiments, OpenAI key is required. Replace a valid OpenAI key in the following line in generate_answers.py:
```
openai.api_key = ""
```

## Running Experiments



