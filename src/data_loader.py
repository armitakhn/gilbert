import os
import pandas as pd


class DataLoader:

    def __init__(self, data_dir):
        self.DATA_DIR = data_dir
        self.PATH_REL_TEXT = os.path.join(self.DATA_DIR + 'relation2text.txt')
        self.PATH_ENT_TEXT = os.path.join(self.DATA_DIR + 'entity2textshort.txt')
        self.base_column_names_ = ['subject', 'property', 'object']
        self.dev_test_column_names_ = self.base_column_names_ + ['y']
        self.train_data = None
        self.dev_data = None
        self.test_data = None

    def get_train_data(self, enrich_train=False):
        train_data = pd.read_csv(os.path.join(self.DATA_DIR + 'train.tsv'), sep='\t', names=self.base_column_names_)
        train_data = self.clean_data_replace_file(train_data, self.base_column_names_)
        if enrich_train:  # TODO: fix this.

            train_data_rev = train_data.copy()
            train_data_rev = train_data_rev.rename(columns={'subject': 'object', 'object': 'subject'})
            train_data_rev = train_data_rev[['subject', 'property', 'object']]
            train_data_rev['property'] = train_data_rev['property'].apply(lambda x: x + ' reverse')
            train_data = pd.concat([train_data, train_data_rev])
            train_data = train_data.reset_index(drop=True)
        return train_data

    def get_dev_data(self):
        dev_data = pd.read_csv(os.path.join(self.DATA_DIR + 'dev.tsv'), sep='\t', names=self.dev_test_column_names_)
        dev_data = self.clean_data_replace_file(dev_data, self.base_column_names_)
        return dev_data

    def get_test_data(self):
        test_data = pd.read_csv(os.path.join(self.DATA_DIR + 'test.tsv'), sep='\t', names=self.dev_test_column_names_)
        test_data = self.clean_data_replace_file(test_data, self.base_column_names_)
        return test_data

    def dict_clean_given_path(self, path):
        clean_dict = {}
        with open(path) as file:
            for line in file:
                key, value = line.strip().split("\t")
                clean_dict[key] = value
        return clean_dict

    def clean_data_replace_file(self, df, rows_apply):
        clean_dict_rel_ent = self.dict_clean_given_path(self.PATH_REL_TEXT)
        clean_dict_rel_ent.update(self.dict_clean_given_path(self.PATH_ENT_TEXT))

        df_ = df.copy()
        df_[rows_apply] = df_[rows_apply].applymap(
            lambda x: clean_dict_rel_ent[x] if x in clean_dict_rel_ent else print("not in the key:{}".format(x)))
        return df_

    def start(self, enrich_train=False):

        self.train_data = self.get_train_data(enrich_train)
        self.dev_data = self.get_dev_data()
        self.test_data = self.get_test_data()

        '''
        self.test_data = self.get_test_data()[0:200]
        self.train_data = self.test_data[['subject', 'property', 'object']].copy()
        self.dev_data = self.test_data[0:5].copy()
        '''
