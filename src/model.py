import numpy as np


class IbmModel1:
    def __init__(self, data_getter):
        self.data_getter = data_getter
        self.t = {}

        self._init_proba()

    def _init_proba(self):
        proba = 1.0/self.data_getter.source_vocabulary_len
        for source_word in self.data_getter.source_vocabulary:
            for target_word in self.data_getter.target_vocabulary:
                key = (target_word, source_word)
                self.t[key] = proba

    def _compute_joint_probability(self, target_sentence, source_sentence, epsilon=1):
        target_sentence_len = len(target_sentence)
        source_sentence_len = len(source_sentence)
        likelihood = 1
        
        for target_word in target_sentence:
            pair_proba = 0
            for source_word in source_sentence:
                pair_proba += self.t[(target_word, source_word)]
            likelihood = pair_proba*likelihood
        
        normalization = epsilon/(source_sentence_len**target_sentence_len)
        return likelihood*normalization

    def _compute_perplexity(self, epsilon=1):
        perplexity = 0
        for sentence_pair in self.data_getter.paralell_sentences:
            target_sentence = sentence_pair[1]
            source_sentence = sentence_pair[0]
            joint_proba = self._compute_joint_probability(target_sentence, source_sentence, epsilon)
            perplexity += np.log2(joint_proba)
            
        return 2.0**(-perplexity)
 