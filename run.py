import os
import argparse
import json
from elm.infer_elm import generate_elm_responses

parser = argparse.ArgumentParser(description='run prompts with elm model.')
parser.add_argument('elm_model_path', help='Path to the elm_model_path')


def get_prompt_config_file(elm_model_path):
    return os.path.join(elm_model_path, "example_prompts.json")

def run(elm_model_path: str):
    prompt_config_file = get_prompt_config_file(elm_model_path)
    
    with open(prompt_config_file, "r") as f:
        prompt_info = json.load(f)
    prompts = [prompt_info["template"].format(input=input) for input in prompt_info["inputs"]]
    print(f"Loaded prompts from: {prompt_config_file}")
    generate_elm_responses(elm_model_path, prompts, verbose=True)
 
if __name__ == "__main__":
    args = parser.parse_args()
    run(args.elm_model_path)