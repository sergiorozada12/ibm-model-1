from src.data import DataGetter
from src.model import IbmModel1

PARALLEL_SENTENCES = [ 
    [ ['das', 'Haus'], ['the', 'house'] ], 
    [ ['das', 'Buch'], ['the', 'book'] ], 
    [ ['ein', 'Buch'], ['a', 'book'] ]
]


if __name__ == "__main__":
    data_getter = DataGetter(PARALLEL_SENTENCES)
    model = IbmModel1(data_getter)
    print(model.t)