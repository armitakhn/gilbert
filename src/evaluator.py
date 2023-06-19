from abc import ABC, abstractmethod
from data_loader import DataLoader
from data_creator import BaseDataCreator
from distance import find_embedding_based_on_hub, find_embedding_based_on_mode, search_in_faiss_pipeline, \
    add_closest_properties
from assets import mapper_index_prop, create_xb_faiss
from sklearn.metrics import accuracy_score
import pandas as pd
from train import ThresholdClassifier
from sklearn.tree import DecisionTreeClassifier

import numpy as np
from filtering import apply_filter_on_pred_props
import tqdm
from scipy import stats
from collections import Counter
import tqdm


def create_df_with_dists_faiss(test_dev_data_embbedded, ent_embeds, col='subject'):
    l_test_dev_embbedded_with_dists = []
    for ent in tqdm.tqdm(set(test_dev_data_embbedded[col])):
        test_ent_emb = test_dev_data_embbedded[test_dev_data_embbedded[col] == ent]

        if ent not in ent_embeds:  # TODO: cold start problem, solve later
            print('cold start problemmmmmmm!!!')
            test_ent_emb["mean_dist"] = 0
            test_ent_emb["min_dist"] = 0
            test_ent_emb["max_dist"] = 0

        D_ent, I_ent = search_in_faiss_pipeline(test_ent_emb, ent_embeds[ent], k=len(ent_embeds[ent]))
        means = [r.mean() for r in D_ent]
        test_ent_emb["mean_dist"] = means
        test_ent_emb["min_dist"] = [r.min() for r in D_ent]
        test_ent_emb["max_dist"] = [r.max() for r in D_ent]
        l_test_dev_embbedded_with_dists.append(test_ent_emb)
    test_dev_embbedded_with_dists = pd.concat(l_test_dev_embbedded_with_dists)
    return test_dev_embbedded_with_dists


def create_closest_props_faiss(test_dev_data_embbedded, prop_embeds, min_k=4):
    test_dev_data_embbedded_c_props = test_dev_data_embbedded.copy()
    test_dev_data_embbedded_c_props['closest_props_ordered'] = None
    mapper_idx_prop = mapper_index_prop(prop_embeds)
    prop_stacks = create_xb_faiss(prop_embeds)

    k = min_k
    found_in_k = pd.Series(test_dev_data_embbedded.shape[0] * [False])
    tmp = test_dev_data_embbedded_c_props

    use_gpu = True
    while True:

        tmp = tmp.loc[~found_in_k]

        _, I_dev_test = search_in_faiss_pipeline(tmp, prop_stacks, k, use_gpu=use_gpu)

        tmp['closest_props_ordered'] = add_closest_properties(I_dev_test, mapper_idx_prop)

        found_in_k = tmp.apply(lambda row: row.property in row.closest_props_ordered, axis=1)
        left = len(found_in_k) - found_in_k.where(found_in_k == True).sum()

        test_dev_data_embbedded_c_props.loc[found_in_k.index] = tmp[found_in_k]

        if left == 0:
            break
        k = k * 4
        if k > 2000:
            use_gpu = False

    return test_dev_data_embbedded_c_props


class BaseEvaluator(ABC):
    def __init__(self, data: DataLoader, model):
        self.train = data.train_data
        self.dev = data.dev_data
        self.test = data.test_data
        self.model = model

    @abstractmethod
    def embedd_train_clusters(self):
        pass

    @abstractmethod
    def embedd_dev_test_data(self):
        pass


class TripleClassifierEvaluator(BaseEvaluator):
    def __init__(self, data: DataLoader, model, logger, cluster_based_on="subject", mode="all", sep_token_bool=False):
        super().__init__(data, model)
        self.cluster_based_on = cluster_based_on
        self.mode = mode
        self.sep_token_bool = sep_token_bool
        self.ent_embeds = None
        self.dev_data_embbedded = None
        self.test_data_embbedded = None
        self.dev_df_with_dists_to_cluster = None
        self.test_df_with_dists_to_cluster = None
        self.best_thr = None
        self.DIST_F = 'mean_dist'  # min or max --> TODO: accept this as input
        self.logger = logger

    def embedd_train_clusters(self):
        self.logger.info('encoding train data clusters..!')
        self.ent_embeds = find_embedding_based_on_hub(self.train, self.model, mode=self.mode,
                                                      hub=self.cluster_based_on)
        self.logger.info('encoding train data done.')

    def embedd_dev_test_data(self):
        self.logger.info('encoding dev data..!')
        self.dev_data_embbedded = find_embedding_based_on_mode(
            self.dev, model=self.model, mode=self.mode, dict_like=self.sep_token_bool)
        self.logger.info('encoding dev data done.')

        self.logger.info('encoding test data..!')
        self.test_data_embbedded = find_embedding_based_on_mode(
            self.test, model=self.model, mode=self.mode, dict_like=self.sep_token_bool)
        self.logger.info('encoding test data done')

    def compute_dist_d_t_to_ent_embedds_faiss(self):
        self.dev_df_with_dists_to_cluster = create_df_with_dists_faiss(self.dev_data_embbedded, self.ent_embeds)
        self.test_df_with_dists_to_cluster = create_df_with_dists_faiss(self.test_data_embbedded, self.ent_embeds)

    def find_thr_dev_train(self):
        X_dev = self.dev_df_with_dists_to_cluster[[self.DIST_F]]
        y_dev = self.dev_df_with_dists_to_cluster[['y']]

        best_sc = 0
        self.logger.info('finding best thr')

        for thr in np.arange(0, 1000, 1):
            m = ThresholdClassifier(thr)

            sc = accuracy_score(y_dev, m.predict(X_dev))
            if best_sc < sc:
                best_sc = sc
                self.best_thr = thr
        self.logger.info(f'best thr: {str(self.best_thr)}')

    def evaluate_on_test(self):
        X_dev = self.dev_df_with_dists_to_cluster[[self.DIST_F]]
        y_dev = self.dev_df_with_dists_to_cluster[['y']]

        X_test = self.test_df_with_dists_to_cluster[[self.DIST_F]]
        y_test = self.test_df_with_dists_to_cluster[['y']]

        thr_cls = ThresholdClassifier(self.best_thr)
        dt = DecisionTreeClassifier(max_depth=1)
        dt.fit(X_dev, y_dev)

        self.logger.info(f'score on test(DecisionTreeClassifier): {accuracy_score(y_test, dt.predict(X_test))}')
        self.logger.info(f'score on val(DecisionTreeClassifier): {accuracy_score(y_dev, dt.predict(X_dev))}')

        self.logger.info(f'score on test: {accuracy_score(y_test, thr_cls.predict(X_test))}')
        self.logger.info(f'score on val: {accuracy_score(y_dev, thr_cls.predict(X_dev))}')


class RelationPredictionEvaluator(BaseEvaluator):
    def __init__(self, data: DataLoader, model, logger, K_NN=200, sep_token_bool=False, sentencebert=True, sp=None):
        super().__init__(data, model)
        self.logger = logger
        self.sep_token_bool = sep_token_bool
        self.K_NN = K_NN
        self.prop_embeds = None
        self.dev_data_embbedded = None
        self.test_data_embbedded = None
        self.dev_data_embbedded_wih_c_props = None
        self.test_data_embbedded_wih_c_props = None
        self.AGG_STRAT = 'k-mode'  # or mean
        self.mode = "so"
        self.sentencebert = sentencebert
        self.sp=sp
        self.train_data_msp = BaseDataCreator(self.train, train=False).map_subj_prop_to_df
        self.dev_data_msp = BaseDataCreator(self.dev, train=False).map_subj_prop_to_df
        self.test_data_msp = BaseDataCreator(self.test, train=False).map_subj_prop_to_df

    def embedd_train_clusters(self):
        self.logger.info('encoding train data property clusters..!')
        self.prop_embeds = find_embedding_based_on_hub(self.train, self.model, mode=self.mode, hub="property",
                                                       dict_like=self.sep_token_bool, sentencebert=self.sentencebert,
                                                       sp=self.sp)

        self.logger.info('encoding train data property clusters done.')

    def embedd_dev_test_data(self):
        self.logger.info('encoding dev data..!')
        self.dev_data_embbedded = find_embedding_based_on_mode(
            self.dev, model=self.model, mode=self.mode, dict_like=self.sep_token_bool, sentencebert=self.sentencebert,
            sp=self.sp)
        self.logger.info('encoding dev data done.')

        self.logger.info('encoding test data..!')
        self.test_data_embbedded = find_embedding_based_on_mode(
            self.test,  model=self.model, mode=self.mode, dict_like=self.sep_token_bool, sentencebert=self.sentencebert,
            sp=self.sp)
        self.logger.info('encoding test data done')

    def compute_min_mode(self, dev_c_props_agg, K, filtering=True):

        agg_k_mode_array = np.array([['None'] * dev_c_props_agg.shape[0]] * len(K), 'O')
        sorted_array = []
        r = 'closest_props_filtered' if filtering else 'closest_props_ordered'
        for i, row in tqdm.tqdm(dev_c_props_agg.iterrows(), total=dev_c_props_agg.shape[0]):
            p_l = row[r].copy()

            d_p = dict(Counter(p_l))
            s_l = [k for k, v in sorted(d_p.items(), key=lambda item: item[1], reverse=True)]
            sorted_array.append(s_l)

            for k, k_mode in enumerate(K):
                agg_k_mode_array[k][i] = stats.mode(p_l[0:k_mode])[0][0]

        return agg_k_mode_array, sorted_array

    def evaluate_dev_best_thr(self, K=[1, 2, 5, 10, 20, 30], filtering=True):

        prec_k_mode = {}

        self.dev_data_embbedded_wih_c_props = create_closest_props_faiss(self.dev_data_embbedded, self.prop_embeds,
                                                                         self.K_NN)

        self.dev_data_embbedded_wih_c_props['closest_props_filtered'] = self.dev_data_embbedded_wih_c_props.apply(
            apply_filter_on_pred_props, map_subj_prop_to_df=self.train_data_msp,
            map_subj_prop_to_df_dev=self.dev_data_msp, map_subj_prop_to_df_test=self.test_data_msp, axis=1)

        dev_c_props_agg = self.dev_data_embbedded_wih_c_props.copy()
        agg_k_mode, sorted_array = self.compute_min_mode(dev_c_props_agg, K, filtering)

        for i, k in enumerate(K):
            dev_c_props_agg["agg_mode_" + str(k)] = agg_k_mode[i]
            prec_k_mode[k] = dev_c_props_agg[(dev_c_props_agg.property == dev_c_props_agg["agg_mode_" + str(k)])].shape[
                                 0] / dev_c_props_agg.shape[0]

        self.best_k_agg = max(prec_k_mode, key=prec_k_mode.get)
        print("best_k_agg", self.best_k_agg)
        print("dict: ", prec_k_mode)
        return sorted_array

    def evaluate_on_test(self, best_dev_k=None, K=[1, 2, 5, 10, 20, 30], filtering=True):
        if not best_dev_k:
            best_dev_k = self.best_k_agg
        prec_k_mode = {}
        self.test_data_embbedded_wih_c_props = create_closest_props_faiss(self.test_data_embbedded, self.prop_embeds,
                                                                          self.K_NN)
        self.test_data_embbedded_wih_c_props['closest_props_filtered'] = self.test_data_embbedded_wih_c_props.apply(
            apply_filter_on_pred_props, map_subj_prop_to_df=self.train_data_msp,
            map_subj_prop_to_df_dev=self.dev_data_msp, map_subj_prop_to_df_test=self.test_data_msp, axis=1)

        train_c_props_agg_full = self.train
        test_c_props_agg_full = self.test_data_embbedded_wih_c_props.copy()

        d = train_c_props_agg_full.groupby(['property']).size().to_frame()
        d = d.reset_index()
        d = d.rename(columns={0: 'count'})
        for jj in [10, 15, 20, 25, 30, 100, 500, 1000, 5000, 10000, 16000]:
            print(jj)
            props_ = list(d[d['count'] < jj].property.values)
            test_c_props_agg = test_c_props_agg_full[test_c_props_agg_full.property.isin(props_)]
            test_c_props_agg = test_c_props_agg.reset_index()

            agg_k_mode, sorted_array = self.compute_min_mode(test_c_props_agg, K, filtering)

            for i, k in enumerate(K):
                test_c_props_agg["agg_mode_" + str(k)] = agg_k_mode[i]
                prec_k_mode[k] = \
                    test_c_props_agg[(test_c_props_agg.property == test_c_props_agg["agg_mode_" + str(k)])].shape[0] / \
                    test_c_props_agg.shape[0]

            best_k_agg = max(prec_k_mode, key=prec_k_mode.get)
            print("best_k_agg", best_k_agg)
            print("dict: ", prec_k_mode)

            test_c_props_agg['sorted_a'] = sorted_array

            def detect_loc(row):
                if row.property == row['agg_mode_20']:
                    return 1
                else:
                    return row.sorted_a.index(row.property) + 1

            test_c_props_agg['MR'] = test_c_props_agg.apply(detect_loc, axis=1)
            test_c_props_agg['MRR'] = 1 / test_c_props_agg['MR']

            self.MR = test_c_props_agg['MR'].sum()/test_c_props_agg['MR'].count()
            self.MRR = test_c_props_agg['MRR'].sum()/test_c_props_agg['MRR'].count()

            print('prec_k_mode[best_dev_k]: ', prec_k_mode[best_dev_k])
            print('self.MR ', self.MR)
            print('self.MRR ', self.MRR)



            #return prec_k_mode[best_dev_k], self.MR, self.MRR
