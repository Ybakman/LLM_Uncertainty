python create_importance_dataset.py --model_name=mistralai/Mistral-7B-v0.1 --device=2 --dataset=trivia_qa --fraction_of_data_to_use='0.5'

python create_importance_dataset.py --model_name=tiiuae/falcon-7b-instruct --device=3 --dataset=trivia_qa --fraction_of_data_to_use='0.5'

python create_importance_dataset.py --model_name=meta-llama/Llama-2-7b-chat-hf --device=4 --dataset=trivia_qa --fraction_of_data_to_use='0.5'

python create_importance_dataset.py --model_name=meta-llama/Llama-2-7b-hf --device=5 --dataset=trivia_qa --fraction_of_data_to_use='0.5'



python create_importance_dataset.py --model_name=mistralai/Mistral-7B-v0.1 --device=6 --dataset=natural_qa --fraction_of_data_to_use='1.0'

python create_importance_dataset.py --model_name=tiiuae/falcon-7b-instruct --device=2 --dataset=natural_qa --fraction_of_data_to_use='1.0'

python create_importance_dataset.py --model_name=meta-llama/Llama-2-7b-chat-hf--device=3 --dataset=natural_qa --fraction_of_data_to_use='1.0'

python create_importance_dataset.py --model_name=meta-llama/Llama-2-7b-hf --device=5 --dataset=natural_qa --fraction_of_data_to_use='1.0'




python create_importance_dataset_word.py --model_name=mistralai/Mistral-7B-v0.1 --device=6 --dataset=trivia_qa --fraction_of_data_to_use='0.5'

python create_importance_dataset_word.py --model_name=tiiuae/falcon-7b-instruct --device=3 --dataset=trivia_qa --fraction_of_data_to_use='0.5'

python create_importance_dataset_word.py --model_name=meta-llama/Llama-2-7b-chat-hf --device=4 --dataset=trivia_qa --fraction_of_data_to_use='0.5'

python create_importance_dataset_word.py --model_name=meta-llama/Llama-2-7b-hf --device=5 --dataset=trivia_qa --fraction_of_data_to_use='0.5'



python create_importance_dataset_word.py --model_name=mistralai/Mistral-7B-v0.1 --device=2 --dataset=natural_qa --fraction_of_data_to_use='1.0'

python create_importance_dataset_word.py --model_name=tiiuae/falcon-7b-instruct --device=3 --dataset=natural_qa --fraction_of_data_to_use='1.0'

python create_importance_dataset_word.py --model_name=meta-llama/Llama-2-7b-chat-hf --device=4 --dataset=natural_qa --fraction_of_data_to_use='1.0'

python create_importance_dataset_word.py --model_name=meta-llama/Llama-2-7b-hf --device=5 --dataset=natural_qa --fraction_of_data_to_use='1.0'



python fine_tune_bert_v1.py --device=5
python fine_tune_bert_v2.py --device=6