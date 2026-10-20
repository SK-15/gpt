import argparse
import glob
import os

import torch

import gpt_v2
from tokenizer import encode, decode, enc

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', default=None, help='default: latest output/*/ckpt.pt')
parser.add_argument('--prompt', default='')
parser.add_argument('--max_new_tokens', type=int, default=500)
parser.add_argument('--num_samples', type=int, default=1)
args = parser.parse_args()

if args.ckpt is None:
    args.ckpt = sorted(glob.glob('output/*/ckpt.pt'))[-1]

device = gpt_v2.device
ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)

# model classes read hyperparams from gpt_v2 globals -> override from checkpoint
for k, v in ckpt['config'].items():
    setattr(gpt_v2, k, v)

model = gpt_v2.GPTLanguageModel().to(device)
model.load_state_dict(ckpt['model'])
model.eval()  # turn off dropout
print(f"loaded {args.ckpt} (step {ckpt['iter']}, val loss {ckpt['val_loss']:.4f})")

# no prompt -> start from <|endoftext|> = fresh story
start_ids = encode(args.prompt) if args.prompt else [enc.eot_token]
context = torch.tensor([start_ids], dtype=torch.long, device=device)

with torch.no_grad():
    for i in range(args.num_samples):
        print(f'--- sample {i+1} ---')
        print(decode(model.generate(context, max_new_tokens=args.max_new_tokens)[0].tolist()))