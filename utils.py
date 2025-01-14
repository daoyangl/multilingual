from baukit import TraceDict, Trace
import torch

def get_activation(model, prompt, device):
    HEADS = [f'model.layers.{i}.self_attn.o_proj' for i in range(model.config.num_hidden_layers)]
    MLPS = [f'model.layers.{i}.mlp' for i in range(model.config.num_hidden_layers)]
    POSTS = [f'model.layers.{i}.post_attention_layernorm' for i in range(model.config.num_hidden_layers)]

    with torch.no_grad():
        prompt = prompt.to(device)
        with TraceDict(model, HEADS+MLPS+POSTS) as ret:
            output = model(prompt, output_hidden_states=True)

        hidden_states = output.hidden_states
        hidden_states = torch.stack(hidden_states, dim=0).squeeze()
        hidden_states = hidden_states.detach().cpu().numpy()
        head_wise_hidden_states = [ret[head].output.squeeze().detach().cpu() for head in HEADS]
        head_wise_hidden_states = torch.stack(head_wise_hidden_states, dim = 0).squeeze().numpy()
        mlp_wise_hidden_states = [ret[mlp].output.squeeze().detach().cpu() for mlp in MLPS]
        mlp_wise_hidden_states = torch.stack(mlp_wise_hidden_states, dim = 0).squeeze().numpy()
        post_wise_hidden_states = [ret[post].output.squeeze().detach().cpu() for post in POSTS]
        post_wise_hidden_states = torch.stack(post_wise_hidden_states, dim = 0).squeeze().numpy()

    return head_wise_hidden_states, mlp_wise_hidden_states, post_wise_hidden_states