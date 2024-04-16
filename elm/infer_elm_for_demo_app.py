# Copyright (c) 2024, SliceX AI, Inc.

from elm.model import *
from elm.utils import batchify
from transformers import AutoTokenizer
import json


def load_elm_model_and_tokenizer(local_path, 
                                 model_config_dict,
                                 device="cuda",
                                 load_partial=True,
                                 get_num_layers_from_ckpt=True):
    """Load ELM model and tokenizer from local checkpoint."""
    model_args = ModelArgs(**model_config_dict)
    model = load_elm_model_from_ckpt(local_path, device=device, model_args=model_args, load_partial=load_partial, get_num_layers_from_ckpt=get_num_layers_from_ckpt)

    tokenizer = AutoTokenizer.from_pretrained(local_path)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    return model, tokenizer


def generate_elm_response_given_model(prompts, model, tokenizer, 
                          device="cuda",
                          max_ctx_word_len=1024,
                          max_ctx_token_len=0,
                          max_new_tokens=500,
                          temperature=0.8, # set to 0 for greedy decoding
                          top_k=200,
                          return_tok_cnt=False,
                          return_gen_only=False,
                          early_stop_on_eos=False):
    """Generate responses from ELM model given an input list of prompts ([str])."""
    if max_ctx_token_len > 0:
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_ctx_token_len).to(device)
    else:
        prompts = [" ".join(p.split(" ")[-max_ctx_word_len:]) for p in prompts]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    
    results = []
    
    input_tok_cnt = torch.numel(inputs.input_ids)

    model.eval()

    out_tok_cnt = 0
    with torch.no_grad():
        temperature = temperature
        top_k = top_k

        outputs = model.generate(inputs.input_ids, max_new_tokens, temperature=temperature, top_k=top_k,
                                 return_gen_only=return_gen_only)

        if return_tok_cnt:
            out_tok_cnt += torch.numel(outputs)

        if early_stop_on_eos:
            mod_outputs = []
            for i in range(len(outputs)):
                curr_out = outputs[i]

                eos_loc_id = -1
                for j in range(len(outputs[i])):
                    tok_id = outputs[i][j]
                    if tok_id == tokenizer.eos_token_id:
                        eos_loc_id = j
                        break
                if eos_loc_id >= 0:
                    curr_out = outputs[i][:eos_loc_id]
                mod_outputs.append(curr_out)
            outputs = mod_outputs
        detokenized_output = tokenizer.batch_decode(outputs, skip_special_tokens=False)

        results = detokenized_output

    if return_tok_cnt:
        return results, (input_tok_cnt, out_tok_cnt)

    return results

def load_elm_model_given_path(elm_model_path, elm_model_config={}, device=None):
    if not device:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Setting device to {device}")
    model_config_dict = {
            "hidden_size": elm_model_config.get("hidden_size", 2048),
            "max_inp_len": elm_model_config.get("max_inp_len", 2048),
            "num_attention_heads": elm_model_config.get("num_attention_heads", 32),
            "num_layers": elm_model_config.get("num_layers", 48),
            "bits": elm_model_config.get("bits", 256),
            "vocab_size": elm_model_config.get("vocab_size", 50304),
            "dropout": elm_model_config.get("dropout", 0.1),
            "use_rotary_embeddings": elm_model_config.get("use_rotary_embeddings", True)
        }
        
    model, tokenizer = load_elm_model_and_tokenizer(local_path=elm_model_path, model_config_dict=model_config_dict, device=device, load_partial=True)
    return {"model": model, "tokenizer": tokenizer}

def generate_elm_responses(elm_model_path, 
                           prompts,
                           device=None, 
                           elm_model_config={},
                           eval_batch_size=1,
                           verbose=True,
                           model_info=None):


    if not device:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Setting device to {device}")

    if not model_info:
        model_info = load_elm_model_given_path(elm_model_path, elm_model_config=elm_model_config, device=device)
    
    model, tokenizer = model_info["model"], model_info["tokenizer"]

    #prompts = [prompt if "[INST]" in prompt else f"[INST]{prompt}[/INST]" for prompt in prompts]
    max_new_tokens = 128
    if "classification" in elm_model_path or "detection" in elm_model_path:
        max_new_tokens = 12
    result = []
    for prompt_batch in batchify(prompts, eval_batch_size):
        responses, _ = generate_elm_response_given_model(prompt_batch,
                                                            model, 
                                                            tokenizer, 
                                                            device=device,
                                                            max_ctx_word_len=1024,
                                                            max_ctx_token_len=512,
                                                            max_new_tokens=max_new_tokens,
                                                            return_tok_cnt=True, 
                                                            return_gen_only=False, 
                                                            temperature=0.0, 
                                                            early_stop_on_eos=True)
    
        for prompt, response in zip(prompt_batch, responses):
            response = response.split("[/INST]")[-1].strip()
            result.append(response)
            if verbose:
                print(json.dumps({"prompt": prompt, "response": response}, indent=4))
                print("\n***\n")
    return result
    
