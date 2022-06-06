import numpy as np


class IbmModel1:
    def __init__(self, data_getter):
        self.data_getter = data_getter
        self.dist_t_s = {}

        self._init_proba()

    def _init_proba(self):
        proba = 1.0/self.data_getter.source_vocabulary_len
        for source_word in self.data_getter.source_vocabulary:
            for target_word in self.data_getter.target_vocabulary:
                key = (target_word, source_word)
                self.dist_t_s[key] = proba

    def _compute_joint_probability(self, target_sentence, source_sentence, epsilon=1):
        target_sentence_len = len(target_sentence)
        source_sentence_len = len(source_sentence)
        likelihood = 1

        for target_word in target_sentence:
            marginal_proba = 0
            for source_word in source_sentence:
                marginal_proba += self.dist_t_s[(target_word, source_word)]
            likelihood = marginal_proba*likelihood

        normalization = epsilon/(source_sentence_len**target_sentence_len)
        return likelihood*normalization

    def _compute_perplexity(self, epsilon=1):
        perplexity = 0
        for source_sentence, target_sentence in self.data_getter.paralell_sentences:
            joint_proba = self._compute_joint_probability(target_sentence, source_sentence, epsilon)
            perplexity += np.log2(joint_proba)

        return 2.0**(-perplexity)

    def train(self, iterations, epsilon):
        self.perplexities = []
        s_total = {}
        for i in range(iterations):
            self.perplexities.append(self._compute_perplexity(epsilon))

            num_occurences = {}
            num_cooccurences = {}

            for source_word in self.data_getter.source_vocabulary:
                num_occurences[source_word] = 0.0
                for target_word in self.data_getter.target_vocabulary:
                    num_cooccurences[(target_word, source_word)] = 0.0

            for source_sentence, target_sentence in self.data_getter.paralell_sentences:
                for target_word in target_sentence:
                    s_total[target_word] = 0.0
                    for source_word in source_sentence:
                        s_total[target_word] += self.dist_t_s[(target_word, source_word)]

                for target_word in target_sentence:
                    for source_word in source_sentence:
                        num_cooccurences[(target_word, source_word)] += self.dist_t_s[(target_word, source_word)]/s_total[target_word]
                        num_occurences[source_word] += self.dist_t_s[(target_word, source_word)]/s_total[target_word]

            for source_word in self.data_getter.source_vocabulary:
                for target_word in self.data_getter.target_vocabulary:
                    self.dist_t_s[(target_word, source_word)] = num_cooccurences[(target_word, source_word)]/num_occurences[source_word]
