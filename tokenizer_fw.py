from tokenizers import Tokenizer

tok = Tokenizer.from_file('data/fwedu_tok16k.json')
vocab_size = tok.get_vocab_size()
eot_token = tok.token_to_id('<|endoftext|>')
encode = lambda s: tok.encode(s).ids
decode = lambda l: tok.decode(l)