from src.data import DataGetter
from src.models import IbmModel1, LanguageModel, Decoder
from src.config import (
    ITERATIONS,
    EPSILON,
    PATH_DATA,
    PATH_TRANSLATION_MODEL,
    PATH_LANGUAGE_MODEL,
    NGRAMS,
    TOP_K,
)


if __name__ == "__main__":
    data_getter = DataGetter(PATH_DATA)

    translation_model = IbmModel1(data_getter, PATH_TRANSLATION_MODEL)
    translation_model.train(ITERATIONS, EPSILON)

    language_model = LanguageModel(data_getter, PATH_LANGUAGE_MODEL, NGRAMS)
    language_model.train()

    decoder = Decoder(language_model, translation_model, TOP_K)
    phrase = decoder.decode(['estic', 'tip', 'que', 'em', 'sermonegi', 'constantment'])
    print(phrase)
