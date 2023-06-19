import tqdm
import random
import pandas as pd
import numpy as np

def correlation_prop_obj(map_ent_prop_to_df, ent_pos):
    dict_correlation_prop_obj = {}
    for ent in map_ent_prop_to_df:
        for prop in map_ent_prop_to_df[ent]:
            if prop not in dict_correlation_prop_obj:
                dict_correlation_prop_obj[prop] = {}
            vals = set(map_ent_prop_to_df[ent][prop][ent_pos].values)

            for val in vals:

                if val in dict_correlation_prop_obj[prop]:
                    dict_correlation_prop_obj[prop][val] |= vals
                else:
                    dict_correlation_prop_obj[prop][val] = vals

    return dict_correlation_prop_obj


def create_possible_ent_false_neg(map_prop_to_df, map_ent_prop_to_df, ent_pos="object"):
    possible_ent_false_neg_dict = {}
    for ent in tqdm.tqdm(map_ent_prop_to_df.keys()):
        possible_ent_false_neg_dict[ent] = {}
        for prop in map_ent_prop_to_df[ent].keys():
            res = set(map_prop_to_df[prop][ent_pos].values)
            res -= set(map_ent_prop_to_df[ent][prop][ent_pos].values)

            possible_ent_false_neg_dict[ent][prop] = list(res)
    return possible_ent_false_neg_dict


def create_false_neg_based_on_ent_row(possible_ent_false_neg_dict, subj, prop, obj, ent_hubs, number_of_false_neg,
                                      add_s_p_o_randomly, textualsequence):
    neg = []
    ent = subj if ent_hubs == "subject" else obj
    choose_from = possible_ent_false_neg_dict[ent][prop]
    random.shuffle(choose_from)

    if len(choose_from) > 0:

        neg_ents_samples = choose_from[0:number_of_false_neg]
        number_of_false_neg_tmp = len(neg_ents_samples)

        for negents in neg_ents_samples:
            if add_s_p_o_randomly and bool(random.getrandbits(1)):
                neg.extend(textualsequence.create_textual_sequence(subj, prop, negents))
            else:
                if ent_hubs == "subject":
                    neg.extend(textualsequence.create_textual_sequence(prop, negents))
                else:
                    neg.extend(textualsequence.create_textual_sequence(negents, prop))
    else:
        number_of_false_neg_tmp = 0

    neg_total = number_of_false_neg_tmp
    return neg, neg_total


def create_true_neg_based_on_ent_row(subj, prop, obj, map_prop_to_df, sorted_dist_prop,
                                     number_of_true_neg, textualsequence):
    ## this function will not be used. we only stick to creating fake negatives for hubs based on entity in the end.
    neg = []

    def create_df_based_on_prop_to_sample_from(props_dict_prop_to_proba, map_prop_to_df):
        final_df_l = []
        chosen_props = random.choices(list(props_dict_prop_to_proba.keys()), props_dict_prop_to_proba.values(), k=5)
        for prop in chosen_props:
            final_df_l.append(map_prop_to_df[prop].sample(1))
        return pd.concat(final_df_l)

    ##### TRUE NEGATIVE ######
    props = sorted_dist_prop[prop]

    df_neg = create_df_based_on_prop_to_sample_from(props, map_prop_to_df)

    if df_neg.shape[0] > 0:
        try:
            df_sample_true_neg = df_neg.sample(number_of_true_neg)
            number_of_true_neg_tmp = number_of_true_neg
        except:
            df_sample_true_neg = df_neg.copy()  # less samples than number of true neg

            number_of_true_neg_tmp = df_sample_true_neg.shape[0]

        for _, row in df_sample_true_neg.iterrows():
            neg.extend(textualsequence.create_textual_sequence(row.subject, row.property, row.object))
    else:
        number_of_true_neg_tmp = 0
    return neg, number_of_true_neg_tmp





def create_pos_based_on_ent_row(map_ent_to_df, subj, prop, obj, ent_hubs, neg_total, add_s_p_o_randomly,
                                textualsequence):
    # ent = subj if ent_hubs == "subject" else obj
    if ent_hubs == "subject":
        ent, ent_comp, ent_comp_ = subj, obj, "object"
    else:
        ent, ent_comp, ent_comp_ = obj, subj, "subject"

    ent_df = map_ent_to_df[ent]

    idx_rmv = ent_df[ent_df.property == prop][ent_df[ent_comp_] == ent_comp].index
    ent_df = ent_df[~ent_df.index.isin(idx_rmv)]  ## fix this line

    if ent_df.shape[0] > 0:

        try:
            df_sample_pos = ent_df.sample(neg_total)
        except:
            df_sample_pos = ent_df.sample(neg_total, replace=True)
        pos = []

        for i, row in df_sample_pos.iterrows():

            if add_s_p_o_randomly and bool(random.getrandbits(1)):
                pos.extend(textualsequence.create_textual_sequence(row.subject, row.property, row.object))
            else:
                if ent_hubs == "subject":
                    pos.extend(textualsequence.create_textual_sequence(row.property, row.object))
                else:
                    pos.extend(textualsequence.create_textual_sequence(row.subject, row.property))
    else:

        pos = []
        for _ in range(neg_total):
            if add_s_p_o_randomly and bool(random.getrandbits(1)):
                pos.extend(textualsequence.create_textual_sequence(subj, prop, obj))
            else:
                if ent_hubs == "subject":
                    pos.extend(textualsequence.create_textual_sequence(prop, obj))
                else:
                    pos.extend(textualsequence.create_textual_sequence(subj, prop))
    return pos


def create_anchor_based_on_ent_row(subj, prop, obj, ent_hubs, neg_total, add_s_p_o_randomly, textualsequence):
    anchors = []
    for _ in range(neg_total):
        if add_s_p_o_randomly and bool(random.getrandbits(1)):
            anchors.extend(textualsequence.create_textual_sequence(subj, prop, obj))
        else:
            if ent_hubs == "subject":
                anchors.extend(textualsequence.create_textual_sequence(prop, obj))
            else:
                anchors.extend(textualsequence.create_textual_sequence(subj, prop))
    return anchors


def create_true_neg_based_on_rel_row(prop, map_prop_to_df, tn, sorted_dist_prop, textual_sequence, choose_based_on_perc=True):

    props = sorted_dist_prop[prop]

    # s,o (with p)   s2,o2(with p)  s3,o3 (with p')
    def create_df_based_on_prop_to_sample_from(props_dict_prop_to_proba, map_prop_to_df):
        final_df_l = []
        chosen_props = random.choices(list(props_dict_prop_to_proba.keys()), props_dict_prop_to_proba.values(), k=5)

        for prop in chosen_props:
            final_df_l.append(map_prop_to_df[prop].sample(1))

        return pd.concat(final_df_l)

    if choose_based_on_perc:
        df_neg = create_df_based_on_prop_to_sample_from(props, map_prop_to_df)

    else:
        negs_p = [p[0] for p in props[0:5]]
        list_df_neg = [map_prop_to_df[prop] for prop in negs_p]

        df_neg = pd.concat(list_df_neg)

    if df_neg.shape[0] > 0:
        try:
            df_sample_true_neg = df_neg.sample(tn)
            number_of_true_neg_tmp = tn

        except:
            df_sample_true_neg = df_neg.copy()  # less samples than number of true neg
            number_of_true_neg_tmp = df_sample_true_neg.shape[0] #take all that is possible

        neg = []
        for _, row in df_sample_true_neg.iterrows():
            neg.extend(textual_sequence.create_textual_sequence(row.subject, row.object))
        # the subject can be later chosen such that they have more similarity with subj
    else:
        number_of_true_neg_tmp = 0

    neg_total = number_of_true_neg_tmp
    return neg, neg_total


def create_pos_based_on_rel_row(subj, prop, obj, map_prop_to_df, neg_total, textual_sequence):

    prop_df = map_prop_to_df[prop]
    prop_df = prop_df[prop_df.subject != subj]
    prop_df = prop_df[prop_df.object != obj]

    pos = []
    if prop_df.shape[0] > 0:
        try:
            df_sample_pos = prop_df.sample(neg_total)
        except:
            df_sample_pos = prop_df.sample(neg_total, replace=True) # choose tekrari

        for _, row in df_sample_pos.iterrows():
            pos.extend(textual_sequence.create_textual_sequence(row.subject, row.object))
    else:
        for _ in range(neg_total):
            pos.extend(textual_sequence.create_textual_sequence(subj, obj)) #same as anchor
    return pos


def create_anchor_based_on_rel_row(subj, obj, neg_total, textual_sequence):
    anchors = []
    for _ in range(neg_total):
        anchors.extend(textual_sequence.create_textual_sequence(subj, obj))
    return anchors



def mapper_index_prop(props_embeds):
    mapper_idx_prop ={}
    i = 0
    for prop, val in props_embeds.items():
        for idx in range(i, i+len(val)):
            mapper_idx_prop[idx] = prop
        i += len(val)
    return mapper_idx_prop

def create_xb_faiss(ent_embeds):
    to_be_stack = []
    for ent in tqdm.tqdm(ent_embeds.keys()):
        to_be_stack.append(np.stack(list(ent_embeds[ent])))

    ent_stacks = np.vstack(to_be_stack)
    return ent_stacks

class TextualSequence:
    def __init__(self, dict_like: bool, ent_abstract: bool, ent_dict_abstract=None):
        self.DICT_LIKE = dict_like
        self.ENT_ABSTRACT = ent_abstract
        self.ENT_DICT_ABSTRACT = ent_dict_abstract

    @staticmethod
    def randp(perc=50):
        return random.randrange(100) < perc

    def create_textual_sequence(self, *args):
        if 1 < len(args) < 4:
            if not self.DICT_LIKE:
                if not self.ENT_ABSTRACT:
                    return [" ".join(args)]
                else:
                    randon_args = []
                    for arg in args:
                        if self.randp(20) and arg in self.ENT_DICT_ABSTRACT:
                            randon_args.append(self.ENT_DICT_ABSTRACT[arg])
                        else:
                            randon_args.append(arg)
                    return [" ".join(randon_args)]
            elif self.DICT_LIKE:
                res_dict = {}
                for i, arg in enumerate(args):
                    key = 'sent_' + str(i)  # the key to the dict is not important at all.
                    if not self.ENT_ABSTRACT:
                        res_dict[key] = arg
                    else:
                        if self.randp(20) and arg in self.ENT_DICT_ABSTRACT:
                            res_dict[key] = self.ENT_DICT_ABSTRACT[arg]
                        else:
                            res_dict[key] = arg

                return [res_dict]
        else:
            raise ValueError("too many arguments given..")



