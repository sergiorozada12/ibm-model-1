import re
import pandas as pd


class DataGetter:
    def __init__(self, path_data):
        self.paralell_sentences = []
        self._prepare_dataset(path_data)

        self.source_vocabulary = []
        self.target_vocabulary = []
        self._prepare_vocabulary()

    def _preprocess_text(self, sentence):
        sentence = re.sub(r'\W+', ' ', sentence)
        sentence = re.sub(' +', ' ', sentence)
        return sentence.lower()

    def _prepare_dataset(self, path_data):
        df = pd.read_csv(path_data, sep='\t', header=None)
        for _, row in df.iterrows():
            sentence_source = self._preprocess_text(row[1])
            sentence_target = self._preprocess_text(row[3])

            self.paralell_sentences.append([sentence_source.split(), sentence_target.split()])

    def _prepare_vocabulary(self):
        for source_sentence, target_sentence in self.paralell_sentences:
            for source_word in source_sentence:
                self.source_vocabulary.append(source_word)
            for target_word in target_sentence:
                self.target_vocabulary.append(target_word)

        self.source_vocabulary = sorted(list(set(self.source_vocabulary)), key=lambda s: s.lower())
        self.target_vocabulary = sorted(list(set(self.target_vocabulary)), key=lambda s: s.lower())

        self.source_vocabulary_len = len(self.source_vocabulary)
        self.target_vocabulary_len = len(self.target_vocabulary)
