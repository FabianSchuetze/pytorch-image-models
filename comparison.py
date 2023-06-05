from collections import defaultdict
from functools import partial

import torch
from torch import nn
import timm

#from vit import Int8Attention, Int8Block
from dataloader import load_val_dataset

#from torch_int.nn.linear import W8A8B8O8Linear

# Load the model
# Get the first layer
# Load a input data
# Get the output
# Compare with quantized module

@torch.no_grad()
def smooth_ln_fcs(ln, fcs, act_scales, alpha=0.5):
    """Takes the original layers in (in fp16 format) and adjustes the ln and
    linear layers. Returns still an fp16 format.

    In evaluation, the results should probably be unchanged?
    """
    if not isinstance(fcs, list):
        fcs = [fcs]
    assert isinstance(ln, nn.LayerNorm)
    for fc in fcs:
        assert isinstance(fc, nn.Linear)
        assert ln.weight.numel() == fc.in_features == act_scales.numel()

    device, dtype = fcs[0].weight.device, fcs[0].weight.dtype
    act_scales = act_scales.to(device=device, dtype=dtype)
    weight_scales = torch.cat([fc.weight.abs().max(
        dim=0, keepdim=True)[0] for fc in fcs], dim=0)
    weight_scales = weight_scales.max(dim=0)[0].clamp(min=1e-5)
    #has the shape out in_features

    scales = (act_scales.pow(alpha) / weight_scales.pow(1-alpha)
              ).clamp(min=1e-5).to(device).to(dtype)

    ln.weight.div_(scales)
    ln.bias.div_(scales)

    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))

def load():
    model = timm.create_model('vit_large_patch16_224', pretrained=True)
    #model = model.to('cuda')
    #block = model.backbone.net.blocks[0]
    #x = torch.load('/tmp/x.pt')
    return model, x


def get_quantized_weights(model, dataset):
    model.eval()
    act_dict = defaultdict(dict)

    def stat_io_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        if name not in act_dict or "input" not in act_dict[name]:
            act_dict[name]["input"] = x.detach().abs().max().item()
        else:
            act_dict[name]["input"] = max(
                act_dict[name]["input"], x.detach().abs().max().item())
        if isinstance(y, tuple):
            y = y[0]
        if name not in act_dict or "output" not in act_dict[name]:
            act_dict[name]["output"] = y.detach().abs().max().item()
        else:
            act_dict[name]["output"] = max(
                act_dict[name]["output"], y.detach().abs().max().item())
    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Linear):
            hooks.append(m.register_forward_hook(
                partial(stat_io_hook, name=name)))
    for data in dataset:
        _ = model(data)  # add evaluation component
    for hook in hooks:
        hook.remove()
    layer_scales = []
    for idx in range(model.config.num_hidden_layers):
        scale_dict = {}
        scale_dict["attn_input_scale"] = act_dict[
            f"model.decoder.layers.{idx}.self_attn.q_proj"]['input'] / 127
        scale_dict["q_output_scale"] = act_dict[
            f"model.decoder.layers.{idx}.self_attn.q_proj"]['output'] / 127
        scale_dict["k_output_scale"] = act_dict[
            f"model.decoder.layers.{idx}.self_attn.k_proj"]['output'] / 127
        scale_dict["v_output_scale"] = act_dict[
            f"model.decoder.layers.{idx}.self_attn.v_proj"]['output'] / 127
        scale_dict["out_input_scale"] = act_dict[
            f"model.decoder.layers.{idx}.self_attn.out_proj"]['input'] / 127
        scale_dict["fc1_input_scale"] = act_dict[
            f"model.decoder.layers.{idx}.fc1"]['input'] / 127
        scale_dict["fc2_input_scale"] = act_dict[
            f"model.decoder.layers.{idx}.fc2"]["input"] / 127
        layer_scales.append(scale_dict)
    return layer_scales




def smooth_module(module, x):
    alpha = 0.5
    module.eval()
    act_dict = {}
    for n, m in module.named_modules():
        if isinstance(m, torch.nn.Linear):
            m.register_forward_hook(
                partial(store_act, act_dict=act_dict, name=n))
    for n, m in module.named_modules():
        if isinstance(m, torch.nn.Linear):
            m.register_forward_hook(
                partial(store_act, act_dict=act_dict, name=n))
    # x_scale = x.abs().max() / 127
    module(x)
    qkv_input = act_dict['attn.qkv'][0]
    qkv_input = qkv_input.view(-1, 1024).abs().detach()
    qkv_input = torch.max(qkv_input, dim=0)[0]
    attn_ln = module.norm1
    qkv = module.attn.qkv
    smooth_ln_fcs(attn_ln, qkv, qkv_input, alpha)
    ffn_ln = module.norm2
    fc1 = module.mlp.fc1
    fc1_input_scales = act_dict['mlp.fc1'][0]
    fc1_input_scales = fc1_input_scales.view(-1, 1024).abs().detach()
    fc1_input_scales= torch.max(fc1_input_scales, dim=0)[0]
    smooth_ln_fcs(ffn_ln, fc1, fc1_input_scales, alpha)


def generate_activation_scales(module, dataset):
    module.eval()
    act_dict = {}
    for n, m in module.named_modules():
        if isinstance(m, torch.nn.Linear):
            m.register_forward_hook(
                partial(store_act, act_dict=act_dict, name=n))
    x_scale = x.abs().max() / 127
    for data in dataset:
        with torch.no_grad():
            y = module(data)
    qkv = act_dict['attn.qkv'][1]
    out_features = qkv.shape[-1] // 3
    q_output_scale = qkv[:, :, :, :out_features].abs().max() / 127
    k_output_scale = qkv[:, :, :, out_features: 2 * out_features].abs().max() / 127
    v_output_scale = qkv[:, :, :, 2 * out_features: ].abs().max() / 127
    proj_input_scale = act_dict['attn.proj'][0].abs().max() / 127
    breakpoint()
    # fc1_input_scale = 1.0
    # fc2_input_scale = 1.0
    fc1_input_scale = act_dict['mlp.fc1'][0].abs().max() / 127
    fc2_input_scale = act_dict['mlp.fc2'][0].abs().max() / 127
    block = Int8Block.from_float(module,
            attn_input_scale=x_scale,
            q_output_scale=q_output_scale,
            k_output_scale=k_output_scale,
            v_output_scale=v_output_scale,
            proj_input_scale=proj_input_scale,
            fc1_input_scale=fc1_input_scale,
            fc2_input_scale=fc2_input_scale)
    # q_x = (x / x_scale).round().to(torch.int8).to('cuda')
    y_hat = block(x)
    breakpoint()
    diff = y - y_hat
    return diff

# def compare(attention_orig, attention_new, x):
    # y_orig = attention_orig(x)
    # y = attention_new(x)
    # return y_orig, y

def main():
    breakpoint()
    model = load()
    dataset = load_val_dataset()
    get_quantized_weights(model, dataset)

    # smooth_module(orig_block,x )
    # generate_quantized_module(orig_block, x)


if __name__ == "__main__":
    main()
