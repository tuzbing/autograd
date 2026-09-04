import tiktoken

class Tokenizer:
	def __init__(self, encoding_name="gpt2"):
		self.enc = tiktoken.get_encoding(encoding_name)
		self.bos_id = self.enc.eot_token
		self.vocab_size = self.enc.n_vocab

	def encode(self, text):
		return self.enc.encode(text, allowed_special={"<|endoftext|>"})

	def decode(self, tokens):
		return self.enc.decode(tokens)