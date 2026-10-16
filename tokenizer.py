import tiktoken

enc = tiktoken.get_encoding("gpt2")
vocab_size = enc.n_vocab #50257
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)

