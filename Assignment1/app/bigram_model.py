# app/bigram_model.py

from collections import defaultdict
import random

class BigramModel:
    def __init__(self, corpus):
        self.bigrams = defaultdict(list)
        self.build_bigrams(corpus)
    
    def build_bigrams(self, corpus):
        for sentence in corpus:
            words = sentence.split()
            for i in range(len(words)-1):
                self.bigrams[words[i]].append(words[i+1])
    
    def generate_text(self, start_word, length):
        word = start_word
        result = [word]
        for _ in range(length-1):
            next_words = self.bigrams.get(word)
            if not next_words:
                break
            word = random.choice(next_words)
            result.append(word)
        return ' '.join(result)
