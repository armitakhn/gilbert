import tqdm
from sentence_transformers import SentenceTransformer, SentencesDataset, losses, models
from sentence_transformers.readers import InputExample
from torch.utils.data import DataLoader
import torch
import os


def raw_to_transformer_triple(triple_sentence_list):
    sentence_transformer_triple = []
    for triple in tqdm.tqdm(triple_sentence_list):
        sentence_transformer_triple.append(InputExample(texts=triple))
    return sentence_transformer_triple


class BaseTripletSbertModelSimple:

    def __init__(self, base_transformer, path, name):
        self.base_transformer = base_transformer
        self.path = path
        self.name = name
        self.model_location = f'{self.path}{self.name}.model'

    def train(self, triples, epoch, batch_size):
        triples = raw_to_transformer_triple(triples)
        model = SentenceTransformer(self.base_transformer)
        train_loss = losses.TripletLoss(model=model)
        train_dataset = SentencesDataset(triples, model)
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

        model.fit(train_objectives=[(train_dataloader, train_loss)],
                  epochs=epoch,
                  warmup_steps=100)

        torch.save(model, self.model_location)
        trained_model = torch.load(self.model_location)
        return trained_model


class BaseTripletSbertModel:
    def __init__(self, base_transformer, path, name, max_seq_length=30, out_dense=256): # TODO add logger to this class
        self.triples = None
        self.base_transformer = base_transformer
        self.path = path
        self.name = name
        self.model_location = f'{self.path}{self.name}.model'
        self.max_seq_length = max_seq_length
        self.out_dense = out_dense
        self.word_embedding_model = models.Transformer(self.base_transformer, max_seq_length=self.max_seq_length)
        self.pooling_model = models.Pooling(self.word_embedding_model.get_word_embedding_dimension())
        self.dense_model = None
        self.model = None

    def init_model(self, dense_last_layer=False):

        modules = [self.word_embedding_model, self.pooling_model]

        if dense_last_layer:
            self.dense_model = models.Dense(in_features=self.pooling_model.get_sentence_embedding_dimension(),
                                            out_features=self.out_dense,
                                            activation_function=torch.nn.Tanh())
            modules.append(self.dense_model)

        self.model = SentenceTransformer(modules=modules)


    def train(self, triples, epoch, warmup_steps=100, triplet_margin=5, batch_size=64):
        triples = raw_to_transformer_triple(triples)
        train_dataset = SentencesDataset(triples, self.model)
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
        train_loss = losses.TripletLoss(model=self.model, triplet_margin=triplet_margin)

        self.model.fit(train_objectives=[(train_dataloader, train_loss)],
                       epochs=epoch,
                       warmup_steps=warmup_steps)

        self.save_trained_model(self.model)
        return self.load_trained_model()

    def save_trained_model(self, model):
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        torch.save(model, self.model_location)

    def load_trained_model(self):
        return torch.load(self.model_location)

class ThresholdClassifier:
    def __init__(self, thr):
        self.thr = thr

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, x_test):
        def compute_score(numb):
            if numb < self.thr:
                return 1
            return -1

        return x_test.applymap(compute_score).values
