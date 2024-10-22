# Grammar-Aligned Decoding

## About

This repository extends the [transformers-CFG](https://github.com/epfl-dlab/transformers-CFG) repository by incorporating support for the **A**daptive **S**ampling with **Ap**proximate expected futures (ASAp) algorithm, which is introduced in the paper [Grammar-Aligned Decoding](https://arxiv.org/abs/2405.21047).

## Installation

Clone the repository:
```
git clone git@github.com:jiayuww/GD.git
```
Create a new Conda environment using the provided requirements file. Replace `/path/to/your/env/gd` with the actual path where you want to store your environment:
```
conda env create -f environment.yml --prefix /path/to/your/env/gd
```

Activate the environment:
```
conda activate /path/to/your/env/gd
```

## Examples

### Inference

```python
from transformers.generation.logits_process import LogitsProcessorList, InfNanRemoveLogitsProcessor
from transformers_gad.grammar_utils import IncrementalGrammarConstraint
from transformers_gad.generation.gad_logits_processor import GrammarAlignedOracleLogitsProcessor

grammar = IncrementalGrammarConstraint(grammar_str, "root", tokenizer)
gad_oracle_processor = GrammarAlignedOracleLogitsProcessor(grammar)
inf_nan_remove_processor = InfNanRemoveLogitsProcessor()
logits_processors = LogitsProcessorList([
    inf_nan_remove_processor,
    gad_oracle_processor,
])

...

for i in range(NUM_ITER):
    output = model.generate(
        input_ids,
        ...,
        logits_processor=logits_processors,
        ...
    )

    # Logit processor must be reset after each generation
    gad_oracle_processor.reset()
```

The ASAp algorithm is implemented as a logit processor. Users can initialize a new `GrammarAlignedOracleLogitsProcessor` for an EBNF grammar and pass it as an argument during generation. Since the logit processor uses an incremental parser internally, users must manually reset the parser state ahead of the next generation the generation.

### Using Trained ASAp Trie

Trained ASAp tries can be saved as a JSON file.

```python
with open(TRIE_PATH, "w") as f:
    f.write(gad_oracle_processor.oracle_trie.json())
```

Saved ASAp tries can be reloaded from a previously saved JSON file and passed during the initialization of the`GrammarAlignedOracleLogitsProcessor`.

```python
from transformers_gad.oracle.oracle_trie import Trie

with open(TRIE_PATH, "r") as f:
    trie = Trie.loads(f.read())

grammar = IncrementalGrammarConstraint(grammar_str, "root", tokenizer)
gad_oracle_processor = GrammarAlignedOracleLogitsProcessor(grammar, trie)
```

Check `scripts/` for more examples.

## Citation

```
@misc{grammaraligneddecoding,
      title={Grammar-Aligned Decoding}, 
      author={Kanghee Park and Jiayu Wang and Taylor Berg-Kirkpatrick and Nadia Polikarpova and Loris D'Antoni},
      year={2024},
      eprint={2405.21047},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2405.21047}, 
}
```

