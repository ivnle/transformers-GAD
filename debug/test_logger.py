import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers_gad.grammar_utils import IncrementalGrammarConstraint
from transformers_gad.generation.logits_process import GrammarConstrainedLogitsProcessor
import argparse
import os
import random
from inference_utils import get_file, load_model_tokenizer_hf
import subprocess
import matplotlib.pyplot as plt
import numpy as np
import get_desired_string_dict
from get_desired_string_dict import stringsofLenk_max, stringsofLenk, convert_grammar
import json
import logging
from tqdm import tqdm
import time
from datetime import datetime
from check_is_valid_string import is_valid_string_start_w_1_all_0, is_valid_string_0, is_valid_string_1, is_valid_string_01
from vllm import LLM, SamplingParams

# import torch
#
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from transformers_gad.grammar_utils import IncrementalGrammarConstraint
# from transformers_gad.generation.logits_process import GrammarConstrainedLogitsProcessor
# import argparse
# import os
# import random
# from inference_utils import get_file, load_model_tokenizer_hf
# import subprocess
# import matplotlib.pyplot as plt
# import numpy as np
# import get_desired_string_dict
# from get_desired_string_dict import stringsofLenk_max, stringsofLenk, convert_grammar
# import json
# import logging
# from tqdm import tqdm
# import time
# from datetime import datetime
# from check_is_valid_string import is_valid_string_start_w_1_all_0, is_valid_string_0, is_valid_string_1, is_valid_string_01
# from vllm import LLM, SamplingParams


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    # Create a file handler that logs even debug messages
    file_handler = logging.FileHandler('/nobackup2/yf/mila/GD/log/debug2.log')
    file_handler.setLevel(logging.INFO)

    # Create a formatter and set it for the handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(file_handler)

    # logger.debug(f"test log")
    logger.info(f"test log info")

    import datetime


    # Function to get current date and time formatted as a string
    # def get_current_time_as_string():
    #     return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    #
    #
    # # Define the log messages with timestamp
    # log_message_debug = f"{get_current_time_as_string()} - DEBUG - test log"
    # log_message_info = f"{get_current_time_as_string()} - INFO - test log info"
    #
    # # Define the path to your log file
    # log_file_path = '/nobackup2/yf/mila/GD/log/debug2.log'
    #
    # # Open the file in append mode ('a') so each log is added to the end of the file
    # with open(log_file_path, 'a') as log_file:
    #     # Write the debug message to the file
    #     log_file.write(f"{log_message_debug}\n")
    #     # Write the info message to the file
    #     log_file.write(f"{log_message_info}\n")

