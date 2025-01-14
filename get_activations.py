import os
import torch
from tqdm import tqdm
import numpy as np
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

from dataset import load_dataset
from utils import get_activation

HF_NAMES = {
    'llama3_8B': 'meta-llama/Meta-Llama-3-8B',
    'llama3_8B_instruct': 'meta-llama/Meta-Llama-3-8B-Instruct',
    'llama3.1_8B': 'meta-llama/Llama-3.1-8B',
    'llama3.1_8B_instruct': 'meta-llama/Llama-3.1-8B-Instruct',
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='llama3_8B')
    parser.add_argument('--dataset_name', type=str, default='cities')
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()

    model_name = HF_NAMES[args.model_name]

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map='auto')
    device = "cuda"

    print("Loading dataset...")
    prompts, labels = load_dataset(args.dataset_name)

    print("Tokenizing prompts...")
    tok_prompts = []
    for prompt in tqdm(prompts):
        tokenized_prompt = tokenizer(prompt, return_tensors='pt').input_ids
        tok_prompts.append(tokenized_prompt)

    all_head_wise_activations = []
    all_mlp_wise_activations = []
    all_post_wise_activations = []
    
    print("Getting activations...")
    for prompt in tqdm(tok_prompts):
        head_wise_activations, mlp_wise_activations, post_wise_activations = get_activation(model, prompt, device)
        all_head_wise_activations.append(head_wise_activations[:,-1,:].copy())
        all_mlp_wise_activations.append(mlp_wise_activations[:,-1,:].copy())
        all_post_wise_activations.append(post_wise_activations[:,-1,:].copy())

    output_dir = f'./features/{args.model_name}/{args.dataset_name}/'
    os.makedirs(output_dir, exist_ok=True)
    
    print("Saving labels...")
    np.save(f'./features/{args.model_name}/{args.dataset_name}/labels.npy', labels)

    print("Saving head wise activations...")
    np.save(f'./features/{args.model_name}/{args.dataset_name}/head_wise.npy', all_head_wise_activations)

    print("Saving mlp wise activations...")
    np.save(f'./features/{args.model_name}/{args.dataset_name}/mlp_wise.npy', all_mlp_wise_activations)

    print("Saving post wise activations...")
    np.save(f'./features/{args.model_name}/{args.dataset_name}/post_wise.npy', all_post_wise_activations)


if __name__=='__main__':
    main()