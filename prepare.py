import numpy as np
from tokenizer import enc

def tokenize_file(src, dst, batch=10_000):
    with open(src, 'r', encoding='utf-8') as f:
        stories = f.read().split('<|endoftext|>')
    eot = enc.eot_token
    total = 0
    with open(dst, 'wb') as out:
        for i in range(0, len(stories), batch):
            chunk = [s.strip() for s in stories[i:i+batch]]
            ids = enc.encode_ordinary_batch(chunk, num_threads=8)
            flat = [t for story in ids for t in (*story, eot)]
            np.array(flat, dtype=np.uint16).tofile(out)  # vocab 50257 < 65535, fits uint16
            total += len(flat)
            print(f'{dst}: {i+len(chunk)}/{len(stories)} stories, {total:,} tokens', flush=True)

tokenize_file('data/TinyStoriesV2-GPT4-train.txt', 'data/train.bin')
tokenize_file('data/TinyStoriesV2-GPT4-valid.txt', 'data/val.bin')