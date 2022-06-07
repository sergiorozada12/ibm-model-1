import json
import numpy as np


class IbmModel1:
    def __init__(self, data_getter, path_model):
        self.data_getter = data_getter
        self.dist_t_s = {}

        self._init_proba()

        self.path_model = path_model

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

        return -perplexity

    def train(self, iterations, epsilon):
        self.perplexities = []
        s_total = {}
        for i in range(iterations):
            perplexity = self._compute_perplexity(epsilon)
            self.perplexities.append(perplexity)

            print(f"Iteration: {i} - Perplexity: {perplexity}")

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

        d = [{'key': k, 'value': v} for k, v in self.dist_t_s.items()]
        with open(self.path_model, 'w') as fp:
            json.dump(d, fp)


class LanguageModel:
    def __init__(self, data_getter, path_model, ngrams):
        self.data_getter = data_getter
        self.ngrams = ngrams
        self.lm = {}

        self.path_model = path_model

    def train(self):
        for sentence in self.data_getter.source_sentences:
            length_sentence = len(sentence)

            for i in range(length_sentence - self.ngrams):
                ngram = sentence[i: i + self.ngrams]
                ngram_key = tuple(ngram[:-1])
                ngram_value = ngram[-1]

                self.lm[ngram_key] = self.lm.get(ngram_key, {})
                self.lm[ngram_key][ngram_value] = 1 + self.lm[ngram_key].get(ngram_value, 0)

            for ngram_key in self.lm.keys():
                total_count = float(sum(self.lm[ngram_key].values()))
                for ngram_value in self.lm[ngram_key].keys():
                    self.lm[ngram_key][ngram_value] /= total_count

        d = [{'key': k, 'value': v} for k, v in self.lm.items()]
        with open(self.path_model, 'w') as fp:
            json.dump(d, fp)

    def generate(self, text):

        sentence_finished = False
        while not sentence_finished:
            th = np.random.random()
            acc = .0

            for word in self.lm[tuple(text[-self.ngrams + 1:])].keys():
                acc += self.lm[tuple(text[-self.ngrams + 1:])][word]
                if acc >= th:
                    text.append(word)
                    break

            if (
                text[-self.ngrams + 1:] == [None, None] or
                acc <= th or
                tuple(text[-self.ngrams + 1:]) not in self.lm
                ):
                sentence_finished = True

        return (' '.join([t for t in text if t]))
