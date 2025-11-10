# enhancing-and-breaking-safeguards-on-open-weight-models

## Setups steps before running:

1. `python -m venv venv`
2. `venv/Scripts/activate` on windows
3. `pip install -r requirements.txt`

## Performing inference on a file:

- Input files should be formatted as json with a list of lists of strings. Each list will become one interaction with a model, and each string should be one prompt to the LLM.
- Get results with this command: `python file_inference.py <model_name> <in_file> <out_file>`
  - Options for 'model_name' are test for a mini model, llama, gpt-oss, or deepseek. Anything else will default to test mini model.
  - The 'in_file' and 'out_files' should both be paths to a json file.
- Output file will be json in the format of a list of strings.

`in_file` example:

```json
[
    ["Translate this to Spanish: 'hello, My name is John'"],
    ["Prompt 1", "Prompt 2"]
]
```

`out_file` example:

```json
[
    "Hola, mi nombre es John",
    "response 2"
]
```

## Getting acceptence statistics

- Run this step with `python acceptance_evaluation.py <in_file> <out_file>`
- Input is expected as a json list of strings in the format of the output from inference
- Output will be a json file with a map of acceptence results
  - If the LLM does not respond in the expected format others will be incremented and the invalid response will be put in the `failed_sentences` list.

`out_file` example:

```json
{
    "refusals": 4,
    "acceptances": 6,
    "others": 1,
    "acceptance_rate": 0.6,
    "failed_sentences": ["LLM did not follow instructions"]
}
```