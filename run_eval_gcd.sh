# baseline, tok_per_sec=34.829
# python eval_gcd.py --min_new_tokens 256 --max_new_tokens 256

# +tensorcores = 33.873 tok/s
# Basically no speed up over baseline. Maybe loading model weights in bf16 makes tensorcores redundant.
# python eval_gcd.py --tc medium

# +compile = 41.198 tok/s
# python eval_gcd.py --compile

# # +compile +tensorcores = 41.247 tok/s
# python eval_gcd.py --compile --tc medium

# # +prefixcache = 34.263 tok/s.
# python eval_gcd.py --use_prefix_cache

# +compile +prefixcache
# BREAKS! Figure out why
# python eval_gcd.py --compile --use_prefix_cache

# +specdecode, tok_per_sec=39.804
# python eval_gcd.py --model_id_assist "meta-llama/Llama-3.2-1B-Instruct" --min_new_tokens 256 --max_new_tokens 256

# +specdecode +compile = 
# BREAKS, "An assistant model is provided, using a dynamic cache instead of a cache of type='static'"
# python eval_gcd.py --model_id_assist "meta-llama/Llama-3.2-1B-Instruct" --min_new_tokens 256 --max_new_tokens 256 --compile

# +gcd = tok_per_sec=20.463
# setting min_new_tokens to a higher value is a problem because we end up forcing generation of something not in the grammar
# python eval_gcd.py --mask gcd --min_new_tokens 1 --max_new_tokens 256

# +gcd +compile, tok_per_sec=22.211
python eval_gcd.py --mask gcd --min_new_tokens 1 --max_new_tokens 256 --compile

# +asap, tok_per_sec=

# +asap +compile, tok_per_sec=