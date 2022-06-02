class DataGetter:
    def __init__(self, paralell_sentences):
        self.paralell_sentences = paralell_sentences

        self.source_vocabulary = []
        self.target_vocabulary = []
        self._prepare_vocabulary()

    def _prepare_vocabulary(self):
        for pair in self.paralell_sentences:
            for source_word in pair[0]: 
                self.source_vocabulary.append(source_word)
            for target_word in pair[1]: 
                self.target_vocabulary.append(target_word)

        self.source_vocabulary = sorted(list(set(self.source_vocabulary)), key=lambda s: s.lower()) 
        self.target_vocabulary = sorted(list(set(self.target_vocabulary)), key=lambda s: s.lower())

        self.source_vocabulary_len = len(self.source_vocabulary)
        self.target_vocabulary_len = len(self.target_vocabulary)
