arg1="$1"
arg2="$2"
arg3="$3"
arg4="$4" 
arg5="$5" 
arg6="$6" 

python generate_answers.py --num_generations_per_prompt='5' --model_name=$arg1 --fraction_of_data_to_use=$arg5 --run_id=$arg2 --temperature=$arg6 --num_beams='1' --top_p='1.0' --device=$arg3 --dataset=$arg4 

python clean_generations.py  --generation_model=$arg1  --run_id=$arg2 

python get_semantic_similarities.py --generation_model=$arg1   --run_id=$arg2 --device=$arg3

python get_likelihoods.py --model_name=$arg1   --device=$arg3  --run_id=$arg2 

python get_uncertainty.py --model_name=$arg1  --run_id=$arg2 

python get_results.py  --model_name=$arg1  --run_id=$arg2 



