import matplotlib.pyplot as plt

from src.data import DataGetter
from src.models import IbmModel1, LanguageModel
from src.config import (
    ITERATIONS,
    EPSILON,
    PATH_DATA,
    PATH_TRANSLATION_MODEL,
    PATH_LANGUAGE_MODEL,
    NGRAMS,
)


if __name__ == "__main__":
    data_getter = DataGetter(PATH_DATA)

    #translation_model = IbmModel1(data_getter, PATH_TRANSLATION_MODEL)
    #translation_model.train(ITERATIONS, EPSILON)

    language_model = LanguageModel(data_getter, PATH_LANGUAGE_MODEL, NGRAMS)
    language_model.train()

    print(language_model.generate(['es', 'va']))

    #plt.plot(translation_model.perplexities)
    #plt.show()
