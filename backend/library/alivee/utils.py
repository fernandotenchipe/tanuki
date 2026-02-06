import random
import hashlib
import time
import numpy as np

def text_to_hash_vector(text, dim=16):
    text = text.lower().strip()
    seed = int(hashlib.sha256(text.encode('utf-8')).hexdigest(), 16) % 10 ** 8
    state = random.getstate()
    random.seed(seed)
    vec = [random.uniform(-1.0, 1.0) for _ in range(dim)]
    random.setstate(state)
    return vec

class CharTokenizer:

    def __init__(self, text_corpus=None, context_size=4):
        self.context_size = context_size
        if text_corpus:
            unique_chars = sorted(list(set(text_corpus)))
            self.vocab = ''.join(unique_chars)
        else:
            self.vocab = 'abcdefghijklmnopqrstuvwxyz .!'
        self.char_to_idx = {c: i for i, c in enumerate(self.vocab)}
        self.idx_to_char = {i: c for i, c in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)

    def encode_char(self, char):
        vec = [0.0] * self.vocab_size
        if char in self.char_to_idx:
            vec[self.char_to_idx[char]] = 1.0
        return vec

    def encode_sequence(self, seq):
        flat = []
        padded = seq.rjust(self.context_size, ' ')[-self.context_size:]
        for char in padded:
            flat.extend(self.encode_char(char))
        return flat

    def decode_output(self, output_vec):
        idx = np.argmax(output_vec)
        return self.idx_to_char.get(idx, '?')

    def generate_dataset(self, text):
        dataset = []
        text = text.lower()
        text = ''.join([c for c in text if c in self.vocab])
        for i in range(len(text) - self.context_size):
            window = text[i:i + self.context_size]
            target_char = text[i + self.context_size]
            inputs = self.encode_sequence(window)
            outputs = self.encode_char(target_char)
            dataset.append({'inputs': inputs, 'outputs': outputs})
        return dataset