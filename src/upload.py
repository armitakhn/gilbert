import pickle
import os
from config import tripledatapath


def store_triples(triples, name):
    if not os.path.exists(tripledatapath):
        os.makedirs(tripledatapath)

    with open(tripledatapath + name + '.pkl', 'wb') as f:
        pickle.dump(triples, f)


