from collections import defaultdict
from functools import partial

import torch
from torch import nn
import timm

from vit import VITINT8
from dataloader import load_val_dataset

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


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
    model = model.to('cuda')
    return model


def evaluation(model, dataset):
    top1 = AverageMeter()
    # breakpoint()
    for idx, data in enumerate(dataset):
        if idx > 101:
            break
        imgs, target = data
        imgs = imgs.to(torch.device('cuda'))
        target = target.to(torch.device('cuda'))
        with torch.no_grad():
            pred_target = model(imgs)  # add evaluation component
        prec1, _ = accuracy(pred_target, target, topk=(1,5))
        top1.update(prec1.data.item(), imgs.size(0))
        if idx % 50 == 0:
            print(f"{idx}: {top1.avg:.3f}")


def get_quantized_weights(model, dataset):
    model.eval()
    act_dict = defaultdict(dict)

    def stat_io_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        if name not in act_dict or "input" not in act_dict[name]:
            # breakpoint()
            if not ('mlp.fc2' in name):
                act_dict[name]["input"] = x.detach().abs().max().item()
            else: # no bitwaste after GELU
                # breakpoint()
                act_dict[name]["input"]= [x.detach().min().item(), x.detach().max().item()]
                # act_dict[name]["input_max"] = x.detach().max().item()
        else:
            if not ('mlp.fc2' in name):
                act_dict[name]["input"] = max(
                    act_dict[name]["input"], x.detach().abs().max().item())
            else:
                # breakpoint()
                tmp_min = x.detach().min().item()
                tmp_max = x.detach().max().item()
                new_min = min(act_dict[name]["input"][0], tmp_min)
                new_max = max(act_dict[name]["input"][1], tmp_max)
                act_dict[name]["input"] = [new_min, new_max]
                    # act_dict[name]["input_min"], tmp_min)
                # act_dict[name]["input_max"] = max(
                    # act_dict[name]["input_max"], tmp_max)

        if isinstance(y, tuple):
            y = y[0]
        if name not in act_dict or "output" not in act_dict[name]:
            if not ('qkv' in name):
                act_dict[name]["output"] = y.detach().abs().max().item()
            else:
                out_features = y.shape[-1] // 3
                q_output = y[:, :, :out_features].abs().max().item()
                k_output = y[:, :, out_features: 2 * out_features].abs().max().item()
                v_output = y[:, :, 2 * out_features:].abs().max().item()
                act_dict[name]['q_output'] = q_output
                act_dict[name]['k_output'] = k_output
                act_dict[name]['v_output'] = v_output
        else:
            if not ('qkv' in name):
                act_dict[name]["output"] = max(
                    act_dict[name]["output"], y.detach().abs().max().item())
            else:
                out_features = y.shape[-1] // 3
                q_output = y[:, :, :out_features].abs().max().item()
                k_output = y[:, :, out_features: 2 * out_features].abs().max().item()
                v_output = y[:, :, 2 * out_features:].abs().max().item()
                act_dict[name]['q_output'] = max(act_dict[name]['q_output'], q_output)
                act_dict[name]['k_output'] = max(act_dict[name]['k_output'], k_output)
                act_dict[name]['v_output'] = max(act_dict[name]['v_output'], v_output)
    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Linear):
            hooks.append(m.register_forward_hook(
                partial(stat_io_hook, name=name)))
    evaluation(model, dataset)
    for hook in hooks:
        hook.remove()
    layer_scales = []
    for idx in range(len(model.blocks)):
        scale_dict = {}
        scale_dict["attn_input_scale"] = act_dict[
            f"blocks.{idx}.attn.qkv"]['input'] / 127
        scale_dict["q_output_scale"] = act_dict[
            f"blocks.{idx}.attn.qkv"]["q_output"] / 127
        scale_dict["k_output_scale"] = act_dict[
            f"blocks.{idx}.attn.qkv"]["k_output"] / 127
        scale_dict["v_output_scale"] = act_dict[
            f"blocks.{idx}.attn.qkv"]["v_output"] / 127
        scale_dict["proj_input_scale"] = act_dict[
            f"blocks.{idx}.attn.proj"]["input"] / 127
        # scale_dict["k_output_scale"] = k_output_scale
        # scale_dict["v_output_scale"] = v_output_scale
        scale_dict["fc1_input_scale"] = act_dict[
            f"blocks.{idx}.mlp.fc1"]['input'] / 127
        fc2_scales = act_dict[f"blocks.{idx}.mlp.fc2"]["input"]
        scale_dict["fc2_input_scale"] = (fc2_scales[1] - fc2_scales[0]) / (2**8 -1)
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
    fc1_input_scales = torch.max(fc1_input_scales, dim=0)[0]
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
    dataset = load_val_dataset()
    model = load()
    # scales = get_quantized_weights(model, dataset)
    # breakpoint()
    scales = torch.load('/tmp/scales.pt')
    int8_model = VITINT8.from_float(model, scales)
    int8_model = int8_model.to(torch.device('cuda'))
    evaluation(int8_model, dataset)

    # smooth_module(orig_block,x )
    # generate_quantized_module(orig_block, x)


if __name__ == "__main__":
    main()
