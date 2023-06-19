import pickle
from config import tripledatapath


def download_triples(name):
    return pickle.load(open(tripledatapath + name + '.pkl', 'rb'))
