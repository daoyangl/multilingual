import torch
import numpy as np
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from einops import rearrange

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
    
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map='auto')
    device = "cuda"

    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads

    labels = np.load(f"./features/{args.model_name}/{args.dataset_name}/labels.npy")
    head_wise_activations = np.load(f"./features/{args.model_name}/{args.dataset_name}/head_wise.npy")
    mlp_wise_activations = np.load(f"./features/{args.model_name}/{args.dataset_name}/mlp_wise.npy")
    post_wise_activations = np.load(f"./features/{args.model_name}/{args.dataset_name}/post_wise.npy")
    print(labels.shape, head_wise_activations.shape, mlp_wise_activations.shape, post_wise_activations.shape)
    head_wise_activations = rearrange(head_wise_activations, 'b l (h d) -> b l h d', h = num_heads)
    print(head_wise_activations.shape)


if __name__=="__main__":
    main()