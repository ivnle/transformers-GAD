import argparse

class ArgumentParser:
    def __init__(self, version="bare"):
        self.version = version
        self.parser = argparse.ArgumentParser(description="Inference argument parser.")

        self._add_basic_arguments()

        # Version-specific arguments
        if version == "gcd":
            self._add_gcd_arguments()
        elif version == "gad":
            self._add_gad_arguments()
        elif version == "bare":
            pass
        else:
            raise ValueError(f"Unknown version: {version}")

    def _add_basic_arguments(self):
        self.parser.add_argument("--model_id", type=str, default="stabilityai/stable-code-instruct-3b",
                                 help="pretrained model checkpoint.")
        self.parser.add_argument("--cache_dir", type=str, default='/staging/groups/dantoni_group/cache_dir/',
                                 help="Where to store cache tokenizers and models.")
        self.parser.add_argument("--num_return_sequences", type=int, default=1,
                                 help="Number of sequences to return.")
        self.parser.add_argument("--repetition_penalty", type=float, default=1.0,
                                 help="Repetition penalty for greedy decoding.")
        self.parser.add_argument("--iter", type=int, default=1,
                                 help="Number of iterations for inference.")
        self.parser.add_argument("--temperature", type=float, default=1.0,
                                 help="Temperature for sampling.")
        self.parser.add_argument("--top_p", type=float, default=1.0,
                                 help="Top p for nucleus sampling.")
        self.parser.add_argument("--top_k", type=int, default=0,
                                 help="Top k for sampling.")
        self.parser.add_argument("--max_new_tokens", type=int, default=512,
                                 help="Maximum number of new tokens to generate.")
        self.parser.add_argument("--device", type=str, choices=["cpu", "cuda", "mps", "xpu", "npu"], default="cpu",
                                 help="The device type.")
        self.parser.add_argument("--dtype", type=str, choices=["float32", "float16", "bfloat16"], default=None,
                                 help="Override the default dtype. If not set, it will use float16 on GPU and float32 on CPU.")
        self.parser.add_argument("--prompt_folder", type=str, default="prompts/SLIA/",
                                 help="Base directory for prompt files.")
        self.parser.add_argument("--output_folder", type=str, default="/staging/groups/dantoni_group/results/SLIA/",
                                 help="Folder to store outputs.")
        self.parser.add_argument("--seed", type=int, default=None,
                                 help="Random seed for reproducibility.")
        self.parser.add_argument("--test_folder", type=str, default="correct/",
                                 help="Directory of test files.")
        self.parser.add_argument("--verbose", action="store_true",
                                 help="Print more information.")
        # self.parser.add_argument("--log_file", type=str, default='log_GAD/track_scores_prob2.log',
        #                     help="Where to store log file.")
        # self.parser.add_argument("--max_length", type=int, default=50,
        #                     help="Maximum length of generated sequences when do not sample.")
        # self.parser.add_argument("--num_beams", type=int, default=5,
        #                     help="Number of beams for beam search.")
        # self.parser.add_argument("--prompt_type", type=str, choices=["bare", "completion", "binary_3"], default="bare",
        #                          help="Prompt type for sygus task. Depreciated, only test pre_prompt for the model.")
        # self.parser.add_argument("--grammar_prompt_file", type=str, default="benchmarks/comp/2018/PBE_BV_Track/PRE_100_10.sl",
        #                          help="File path to prompts for sygus task. Depreciated, only test one grammar prompt for the model.")
        # self.parser.add_argument("--instruct_prompt_file", type=str, default="prompts/pre_prompt.jsonl",
        #                          help="File path to prompts for sygus task. Depreciated, only test one instruct prompt for the model.")
    def _add_gcd_arguments(self):
        self.parser.add_argument("--grammar_folder", type=str, default="examples/sygus/SLIA/",
                                 help="Base directory for grammar constraint files. This directory should be exactly one level above the directory containing the grammar file.")
        # self.parser.add_argument("--grammar_file", type=str, default="find_inv_bare.ebnf",
        #                          help="Grammar file to test, sygus is constructed automatically.")

    def _add_gad_arguments(self):
        self.parser.add_argument("--grammar_folder", type=str, default="examples/sygus/SLIA/",
                                 help="Base directory for grammar constraint files. This directory should be exactly one level above the directory containing the grammar file.")
        # self.parser.add_argument("--grammar_file", type=str, default="find_inv_bare.ebnf",
        #                          help="Grammar file only to test, sygus is constructed automatically.")
        self.parser.add_argument("--trie_folder", type=str, default="/staging/groups/dantoni_group/tries/SLIA/",
                                 help="Folder to store trie files.")

    def parse_args(self):
        return self.parser.parse_args()