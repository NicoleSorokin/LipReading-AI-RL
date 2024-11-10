import numpy as np
import tensorflow as tf
import os

class Align(object):
    def __init__(self, align_path):
        self.entries = []
        self.align_path = align_path

    def load_alignment(self):
        with open(self.align_path, 'r') as file:
            for line in file:
                # Parse each line
                start, end, label = line.strip().split()
                entry = [int(int(start)/1000), int(int(end)/1000), label]
                self.entries.append(entry)

    def get_sentence(self):
        # Return only the tokens, ignore timestamps, ignore pauses
        return " ".join([entry[-1] for entry in self.entries if entry[-1] not in ['sp', 'sil']])
    
    def to_tensor(self):
        vocab = list("abcdefghijklmnopqrstuvwxyz'?!123456789 ")
        char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
        
        sentence = self.get_sentence()  # Get sentence as a string
        char_tensor = tf.strings.unicode_split(sentence, input_encoding='UTF-8')

        return char_to_num(char_tensor)
