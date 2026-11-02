import itertools
import os

import numpy as np
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer, Tokenizer

TOK_PATH = 'data/fwedu_tok16k.json'
TOK_SAMPLE_DOCS = 300_000            # ~1.3GB text to train tokenizer
VAL_DOCS = 10_000                    # ~10M tokens
TRAIN_TOKENS = 1_500_000_000

def docs():
    ds = load_dataset('HuggingFaceFW/fineweb-edu', name='sample-10BT',
                      split='train', streaming=True)
    for row in ds:
        yield row['text']

# 1. train tokenizer
if not os.path.exists(TOK_PATH):
    tok = ByteLevelBPETokenizer()
    tok.train_from_iterator(itertools.islice(docs(), TOK_SAMPLE_DOCS), vocab_size=16384,
                            min_frequency=2, special_tokens=['<|endoftext|>'])
    tok.save(TOK_PATH)

tok = Tokenizer.from_file(TOK_PATH)
eot = tok.token_to_id('<|endoftext|>')

# 2. tokenize into .bin
def flush(out, batch):
    ids = [t for e in tok.encode_batch(batch) for t in (*e.ids, eot)]
    np.array(ids, dtype=np.uint16).tofile(out)   # vocab 16384 fits uint16
    return len(ids)

def write_bin(dst, texts, max_tokens):
    total, batch = 0, []
    with open(dst, 'wb') as out:
        for t in texts:
            batch.append(t)
            if len(batch) < 1000:
                continue
            total += flush(out, batch)
            batch = []
            print(f'{dst}: {total:,} tokens', flush=True)
            if total >= max_tokens:
                return
        if batch:
            total += flush(out, batch)

it = docs()
write_bin('data/fwedu_val.bin', itertools.islice(it, VAL_DOCS), float('inf'))
write_bin('data/fwedu_train.bin', it, TRAIN_TOKENS)