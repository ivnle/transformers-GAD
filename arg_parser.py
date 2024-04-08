import argparse

class ArgumentParser:
    def __init__(self, version="bare"):
        self.version = version
        self.parser = argparse.ArgumentParser(description="Inference parser.")

        # Common arguments across versions
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
        self.parser.add_argument("--model_id", type=str, default="mistralai/Mistral-7B-Instruct-v0.1",
                                 help="pretrained model checkpoint.")
        self.parser.add_argument("--cache_dir", type=str, default='/nobackup2/yf/mila/GD_caches/',
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
        self.parser.add_argument("--prompt_type", type=str, choices=["bare", "completion"], default="bare",
                                 help="Prompt type for sygus task.")
        self.parser.add_argument("--output_folder", type=str, default="/nobackup2/yf/mila/GD/results/",
                                 help="Output folder to store results.")
        self.parser.add_argument("--device", type=str, choices=["cpu", "cuda", "mps", "xpu", "npu"], default="cpu",
                                 help="The device type.")
        self.parser.add_argument("--sygus_prompt_file", type=str, default="/nobackup2/yf/mila/GD/prompts/pre_prompt.jsonl",
                                 help="File path to prompts for sygus task.")
        self.parser.add_argument("--dtype", type=str, choices=["float32", "float16", "bfloat16"], default=None,
                                 help="Override the default dtype. If not set, it will use float16 on GPU and float32 on CPU.")
        # self.parser.add_argument("--prompt", type=str, default=f"Generate a program.",
        #                          help="Depreciated, warning: only test prompt for the model.")
        # self.parser.add_argument("--log_file", type=str, default='/nobackup2/yf/mila/GD/log_GAD/track_scores_prob2.log',
        #                     help="Where to store log file.")
        # self.parser.add_argument("--max_length", type=int, default=50,
        #                     help="Maximum length of generated sequences when do not sample.")
        # self.parser.add_argument("--seed", type=int, default=42,
        #                     help="Random seed for reproducibility.")
        # self.parser.add_argument("--num_beams", type=int, default=5,
        #                     help="Number of beams for beam search.")
    def _add_gcd_arguments(self):
        self.parser.add_argument("--base_grammar_dir", type=str, default="/nobackup2/yf/mila/GD/examples/grammars/",
                                 help="Base directory for test grammars.")
        self.parser.add_argument("--grammar_file", type=str, default="string_01.ebnf",
                                 help="Grammar file to test.")
        self.parser.add_argument("--grammar_name", type=str, default="PRE_100",
                                 help="Name of the grammar, mainly used for call grammar file.")

    def _add_gad_arguments(self):
        self.parser.add_argument("--base_grammar_dir", type=str, default="/nobackup2/yf/mila/GD/examples/grammars/",
                                 help="Base directory for test grammars.")
        self.parser.add_argument("--grammar_file", type=str, default="string_01.ebnf",
                                 help="Grammar file to test.")
        self.parser.add_argument("--grammar_name", type=str, default="PRE_100",
                                 help="Name of the grammar, mainly used for call grammar file.")
        self.parser.add_argument("--trie_folder", type=str, default="/nobackup2/yf/mila/GD/results_trie/",
                                 help="Folder to store trie files.")

    def parse_args(self):
        return self.parser.parse_args()