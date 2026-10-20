import os
import time
from datetime import datetime

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from tokenizer import vocab_size

# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 50
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
# ------------

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+block_size+1].astype(np.int64)) for i in ix])
    x,y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.num_heads = num_heads
        # self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.qkv = nn.Linear(n_embd, 3*num_heads*head_size, bias=False)
        self.proj = nn.Linear(head_size*num_heads,n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):

        B,T,C = x.shape
        q,k,v = self.qkv(x).split(C, dim=2) # each (B, T, C)
        q = q.view(B,T, self.num_heads,-1).transpose(1,2) # (B, nh, T, hs)
        k = k.view(B,T, self.num_heads,-1).transpose(1,2) 
        v = v.view(B,T, self.num_heads,-1).transpose(1,2)
        # softmax(QK^T/sqrt(d)) + causal mask + dropout + @V, in one fused kernel
        out = F.scaled_dot_product_attention(q,k,v, is_causal=True,
                                             dropout_p=dropout if self.training else 0.0)
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # concat heads back
        return self.dropout(self.proj(out)) 

class FeedForward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd,4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd,n_embd),
            nn.Dropout(dropout)
        )

    def forward(self,x):
        out = self.net(x)
        return out


class Block(nn.Module):

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head,head_size)
        self.ff = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2= nn.LayerNorm(n_embd)

    def forward(self,x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.token_embedding_table = nn.Embedding(vocab_size,n_embd)
        self.positional_embedding_table = nn.Embedding(block_size,n_embd)
        self.block = nn.Sequential(*[Block(n_embd,n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd,vocab_size)

        self.apply(self._init_weights)
        self.lm_head.weight = self.token_embedding_table.weight 

    def _init_weights(self,module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self,idx,targets=None):

        B, T = idx.shape

        tok_embd = self.token_embedding_table(idx)
        pos_embd = self.positional_embedding_table(torch.arange(T, device=device))
        x = tok_embd + pos_embd
        x = self.block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits,targets)

        return logits,loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
 
if __name__ == '__main__':
    out_dir = os.path.join('output', datetime.now().strftime('%Y-%m-%d'))
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')

    torch.manual_seed(1337)

    train_data = np.memmap('data/train.bin', dtype=np.uint16, mode='r')
    val_data = np.memmap('data/val.bin', dtype=np.uint16, mode='r')

    model = GPTLanguageModel()
    m = model.to(device)
    # print the number of parameters in the model
    print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    train_log = open(os.path.join(out_dir, 'train_loss.csv'), 'w')
    train_log.write('step,loss\n')
    eval_log = open(os.path.join(out_dir, 'eval_loss.csv'), 'w')
    eval_log.write('step,train_loss,val_loss,elapsed_sec\n')
    start = time.time()

    best_val_loss = float('inf')

    for iter in range(max_iters):

        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}", flush=True)
            eval_log.write(f"{iter},{losses['train']:.4f},{losses['val']:.4f},{time.time()-start:.1f}\n")
            eval_log.flush()

            if losses['val'] < best_val_loss:
                best_val_loss = losses['val'].item()
                torch.save({
                    'model': model.state_dict(),
                    'config': {
                        'vocab_size': vocab_size,
                        'block_size': block_size,
                        'n_embd': n_embd,
                        'n_head': n_head,
                        'n_layer': n_layer,
                        'dropout': dropout,
                    },
                    'iter': iter,
                    'val_loss': best_val_loss,
                }, ckpt_path)
                print(f"saved checkpoint to {ckpt_path}", flush=True)

        # sample a batch of data
        xb, yb = get_batch('train')

        # evaluate the loss
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        train_log.write(f'{iter},{loss.item():.4f}\n')
        if iter % 100 == 0:
            train_log.flush()

    train_log.close()
    eval_log.close()





