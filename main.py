import matplotlib.pyplot as plt

from src.data import DataGetter
from src.model import IbmModel1
from src.config import ITERATIONS, EPSILON, PATH_DATA


if __name__ == "__main__":
    data_getter = DataGetter(PATH_DATA)
    model = IbmModel1(data_getter)
    model.train(ITERATIONS, EPSILON)

    plt.plot(model.perplexities)
    plt.show()
