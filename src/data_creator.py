import tqdm
from assets import create_possible_ent_false_neg, create_false_neg_based_on_ent_row, create_true_neg_based_on_ent_row, \
    TextualSequence, create_pos_based_on_ent_row, create_anchor_based_on_ent_row, create_true_neg_based_on_rel_row, \
    create_pos_based_on_rel_row, create_anchor_based_on_rel_row


class DomainRangeSortedDistProp:
    def __init__(self, train_data, map_prop_to_df, mode, identical):
        self.map_prop_to_df = map_prop_to_df
        self.properties_set = set(train_data.property)
        self.mode = mode
        self.identical = identical
        self.sorted_dist_prop = self.find_closest_property()

    def find_closest_property(self):
        props = list(self.properties_set)
        prop_to_other_prop_share_prec = {}
        for pt1 in tqdm.tqdm(props):
            p1_subj = set(self.map_prop_to_df[pt1].subject)
            p1_obj = set(self.map_prop_to_df[pt1].object)

            tot_subj = len(p1_subj)
            tot_obj = len(p1_obj)

            if pt1 not in prop_to_other_prop_share_prec:
                prop_to_other_prop_share_prec[pt1] = []

            for pt2 in props:
                if pt2 == pt1 and not self.identical:
                    continue
                p2_subj = set(self.map_prop_to_df[pt2].subject)
                p2_obj = set(self.map_prop_to_df[pt2].object)

                l_subj = len(p1_subj.intersection(p2_subj))
                l_obj = len(p1_obj.intersection(p2_obj))

                if self.mode == 'domain':
                    prop_to_other_prop_share_prec[pt1].append(tuple([pt2, l_subj * 100 / tot_subj]))

                if self.mode == 'range':
                    prop_to_other_prop_share_prec[pt1].append(tuple([pt2, l_obj * 100 / tot_obj]))
                if self.mode == 'both':
                    prop_to_other_prop_share_prec[pt1].append(tuple([pt2, l_subj * 100 / tot_subj]))
                    prop_to_other_prop_share_prec[pt1].append(tuple([pt2, l_obj * 100 / tot_obj]))

        if self.mode == 'both':
            for k in prop_to_other_prop_share_prec:
                # sum_imps = 0

                inp = prop_to_other_prop_share_prec[k]

                d = {x: 0 for x, _ in inp}
                for name, num in inp:
                    d[name] += num / 2
                    # sum_imps+= num/2

                # for pp in d:
                #    try:
                #        d[pp] = d[pp]/sum_imps
                #    except:
                #        pass

                prop_to_other_prop_share_prec[k] = list(map(tuple, d.items()))

        prop_to_other_prop_share_prec = {k: sorted(v, key=lambda e: e[1], reverse=True) for k, v in
                                         prop_to_other_prop_share_prec.items()}
        return prop_to_other_prop_share_prec

    def refine_sorted_dist_prop_based_on_threshold(self, thr, min_number_props=10, unified=False) -> dict:
        final_dict = {}

        def num_of_props_higher_treshhold(prop_l_t, thr):
            count = 0
            for pp_t in prop_l_t:
                if pp_t[1] >= thr:
                    count += 1
            return count

        def compute_percentage_sorted_dist_prop(thr, thr_p):
            # dict_how_many = {}
            list_more_than_thr = []
            c = 0
            for key, val in self.sorted_dist_prop.items():
                # dict_how_many[key] = num_of_props_higher_treshhold(val, thr)
                if num_of_props_higher_treshhold(val, thr) >= thr_p:
                    c += 1
                    list_more_than_thr.append(key)

            pc = c / len(self.sorted_dist_prop) * 100
            return pc, list_more_than_thr

        def normalize_perc(list_poss_props_thr, thr):
            new_dict = {}
            to_devide = 0
            for name, num in list_poss_props_thr:
                if num >= thr:
                    to_devide += num
                    new_dict[name] = num
            if to_devide != 0:
                return {k: v / to_devide for k, v in new_dict.items()}
            else:
                return {k: v / 1 for k, v in new_dict.items()}

        pc, list_more_than_thr = compute_percentage_sorted_dist_prop(thr, min_number_props)

        print(
            "at least {pc} percent of the properties have {min_number_props} properties having {thr} percent in common with them".format(
                pc=pc, min_number_props=min_number_props, thr=thr))

        for prop, list_poss_props_thr in self.sorted_dist_prop.items():
            if prop in list_more_than_thr:
                final_dict[prop] = normalize_perc(list_poss_props_thr, thr)
            else:
                # print(list_poss_props_thr[:min_number_props])
                final_dict[prop] = normalize_perc(list_poss_props_thr[:min_number_props], thr=0)

        if unified:
            for prop in final_dict:
                each_score = 100 / len(final_dict[prop])
                for p2 in final_dict[prop]:
                    final_dict[prop][p2] = each_score

        return final_dict




def split_data_based_on_one_column(data, col):

    DICT_COL = {}
    for df_t in tqdm.tqdm(data.groupby([col])):
        dfs = df_t[1]
        DICT_COL[dfs[col].values[0]] = dfs
    return DICT_COL

def split_data_based_on_two_columns(data, col1, col2):

    DICT_COL1_COL2 = {}

    def add_col1_col2_dict(df_t):
        dt = df_t.copy()
        pp = dt[col1].values[0]
        oo = dt[col2].values[0]
        if pp not in DICT_COL1_COL2:
            DICT_COL1_COL2[pp] = {}
        DICT_COL1_COL2[pp][oo] = dt

    for df_t in tqdm.tqdm(data.groupby([col1, col2])):
        add_col1_col2_dict(df_t[1])

    return DICT_COL1_COL2


class BaseDataCreator:

    def __init__(self, data, train=True):
        if train:
            self.map_prop_to_df = split_data_based_on_one_column(data, "property")
            self.map_subj_to_df = split_data_based_on_one_column(data, "subject")
            self.map_obj_to_df = split_data_based_on_one_column(data, "object")
            self.map_prop_obj_to_df = split_data_based_on_two_columns(data, "property", "object")
        self.map_subj_prop_to_df = split_data_based_on_two_columns(data, "subject", "property")


class SentenceTripleEntity(BaseDataCreator):

    def __init__(self, train_data, entity, fn, tn, logger, thr_d_r=0.0001, min_number_props_d_r=1, unified_d_r=True,
                 rnd_spo_bool=False, rnd_abstract_bool=False, sep_token_bool=False):
        """

        :param train_data: training data
        :param entity: entity based on which the clusters are created. "subject" or "object"
        :param fn: number of false negative per anchor, for the entity
        :param tn: number of true negatives per anchor, for the entity
        :param rnd_spo_bool: boolean , if True, randomly creates textual sequences using subj, prop, obj
        :param rnd_abstract_bool: boolean , if True, randomly replaces the entity with its abstract when creating the textual seq
        :param sep_token_bool: if True, data created such that when passed to BERT [SEP] token between two parts
        """
        self.train_data = train_data
        super().__init__(self.train_data)  # self.map_prop_to_df, self.map_subj_to_df ,self.map_prop_obj_to_df ,self.map_subj_prop_to_df
        self.logger = logger
        self.entity_cluster = entity
        self.fn = fn
        self.tn = tn
        self.thr_d_r = thr_d_r
        self.min_number_props_d_r = min_number_props_d_r
        self.unified_d_r = unified_d_r
        self.rnd_spo_bool = rnd_spo_bool
        self.rnd_abstract_bool = rnd_abstract_bool
        self.sep_token_bool = sep_token_bool
        self.possible_ent_false_neg_dict = create_possible_ent_false_neg(self.map_prop_to_df, self.map_subj_prop_to_df,
                                                                         ent_pos="object")  # TODO: based on subject does not work
        if self.entity_cluster == 'subject':
            self.map_ent_to_df = self.map_subj_to_df
        elif self.entity_cluster == 'object':
            self.map_ent_to_df = self.map_object_to_df
        else:
            raise ValueError(f'entity cannot be {self.entity_cluster} only subject or object is valid')

        self.textualsequence = TextualSequence(self.sep_token_bool, self.rnd_abstract_bool)
        self.domainrangesorteddistprop = DomainRangeSortedDistProp(self.train_data, self.map_prop_to_df, mode='both',
                                                                   identical=False)
        self.sorted_dist_prop = self.domainrangesorteddistprop. \
            refine_sorted_dist_prop_based_on_threshold(thr=self.thr_d_r,
                                                       min_number_props=self.min_number_props_d_r,
                                                       unified=self.unified_d_r)

        self.sentences_triple = []

    def create_sentences_triple(self) -> list:

        # anchor positive negative for each row that comes, number of neg maximum

        for k, row in tqdm.tqdm(self.train_data.iterrows(), total=self.train_data.shape[0]):
            subj, prop, obj = row

            #### NEG #### po/spo that belongs to a different subject than subj. ex: row: BO marriedTo MO --> marriedTo MT
            neg_fn, neg_total_fn = create_false_neg_based_on_ent_row(self.possible_ent_false_neg_dict, subj, prop, obj,
                                                                     self.entity_cluster,
                                                                     self.fn, self.rnd_spo_bool, self.textualsequence)
            ##### TRUE NEG ###########
            neg_tn, neg_total_tn = create_true_neg_based_on_ent_row(subj, prop, obj, self.map_prop_to_df,
                                                                    self.sorted_dist_prop,
                                                                    self.tn, self.textualsequence)
            neg = neg_fn
            neg.extend(neg_tn)
            neg_total = neg_total_fn + neg_total_tn

            if neg_total == 0:
                continue

            ##### POSITIVE ###### for the same ent, choose another partial fact belonging to ent if possible
            pos = create_pos_based_on_ent_row(self.map_ent_to_df, subj, prop, obj, self.entity_cluster, neg_total,
                                              self.rnd_spo_bool, self.textualsequence)

            ##### ANCHOR ###### for the same ent, and row, create anchor
            anchors = create_anchor_based_on_ent_row(subj, prop, obj, self.entity_cluster, neg_total, self.rnd_spo_bool,
                                                     self.textualsequence)

            for i in range(len(anchors)):
                self.sentences_triple.append([anchors[i], pos[i], neg[i]])

        number_successful_spo = 0
        if self.rnd_spo_bool and self.entity_cluster == "subject":
            if len(self.possible_ent_false_neg_dict[subj][prop]) > 0:
                self.sentences_triple.append(
                    [self.textualsequence.create_textual_sequence(subj, prop, obj)[0],
                     self.textualsequence.create_textual_sequence(prop, obj)[0],
                     self.textualsequence.create_textual_sequence(prop, self.possible_ent_false_neg_dict[subj][prop][0])[0]])
                number_successful_spo += 1


        elif self.rnd_spo_bool and self.entity_cluster == "object":
            if len(self.possible_ent_false_neg_dict[obj][prop]) > 0:
                self.sentences_triple.append(
                    [self.textualsequence.create_textual_sequence(subj, prop, obj)[0],
                     self.textualsequence.create_textual_sequence(subj, prop)[0],
                     self.textualsequence.create_textual_sequence(prop, self.possible_ent_false_neg_dict[obj][prop][0])[
                         0]])
                number_successful_spo += 1

        self.logger.info(f"number_successful_spo: {number_successful_spo}")

        return self.sentences_triple


class SentencePartialFactRelation(BaseDataCreator):

    def __init__(self, train_data, tn, thr_d_r=0.0001, min_number_props_d_r=10, unified_d_r=True,
                 rnd_abstract_bool=False, sep_token_bool=False, choose_based_on_perc=True):
        """
        :param train_data: training data
        :param tn: number of total negatives per anchor, for the entity
        :param rnd_abstract_bool: boolean , if True, randomly replaces the entity with its abstract when creating the textual seq
        :param sep_token_bool: if True, data created such that when passed to BERT [SEP] token between two parts
        """
        self.train_data = train_data
        super().__init__(
            self.train_data)  # self.map_prop_to_df, self.map_subj_to_df ,self.map_prop_obj_to_df ,self.map_subj_prop_to_df

        self.tn = tn
        self.thr_d_r = thr_d_r
        self.min_number_props_d_r = min_number_props_d_r
        self.unified_d_r = unified_d_r
        self.rnd_abstract_bool = rnd_abstract_bool
        self.sep_token_bool = sep_token_bool
        self.choose_based_on_perc = choose_based_on_perc



        self.textualsequence = TextualSequence(self.sep_token_bool, self.rnd_abstract_bool)
        self.domainrangesorteddistprop = DomainRangeSortedDistProp(self.train_data, self.map_prop_to_df, mode='both',
                                                                   identical=False)
        self.sorted_dist_prop = self.domainrangesorteddistprop. \
            refine_sorted_dist_prop_based_on_threshold(thr=self.thr_d_r,
                                                       min_number_props=self.min_number_props_d_r,
                                                       unified=self.unified_d_r)
        self.sentences_PFs = []

    def create_partial_fact_relation(self) -> list:
        for k, row in tqdm.tqdm(self.train_data.iterrows(), total=self.train_data.shape[0]):
            subj, prop, obj = row

            #### Negative
            neg, neg_total = create_true_neg_based_on_rel_row(prop, self.map_prop_to_df, self.tn, self.sorted_dist_prop, self.textualsequence,
                                                              self.choose_based_on_perc)

            #### Positive
            pos = create_pos_based_on_rel_row(subj, prop, obj, self.map_prop_to_df, neg_total, self.textualsequence)

            #### Anchor
            anchors = create_anchor_based_on_rel_row(subj, obj, neg_total, self.textualsequence)


            for i in range(len(anchors)):


                self.sentences_PFs.append([anchors[i], pos[i], neg[i]])

        return self.sentences_PFs

