import tqdm
from sentence_transformers.readers import InputExample
import string
import re
from nltk.corpus import stopwords


# nltk.download('stopwords')

def remove_number(text):
    return ''.join([i for i in text if not i.isdigit()])


def remove_underlines(text):
    return text.replace('_', ' ').replace('-', ' ').replace('–', ' ')


def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))


def trucate_text(text):
    return ' '.join(text.split(' ')[:20])


def apply_strip(text):
    return text.strip()


def remove_stopwords(text):
    pattern = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')
    return pattern.sub('', text)


def ascii_encoding(text):
    return text.encode("ascii", "ignore").decode()


def clean_data(data, rows_apply=['subject', 'property', 'object'], remove_punct=True, remove_numb=True,
               remove_underline=True, trim=True, lower_case=True, remove_stp=False, truncate=False, encode=False):
    df = data.copy()
    if lower_case:
        df[rows_apply] = df[rows_apply].applymap(str.lower)
    if encode:
        df[rows_apply] = df[rows_apply].applymap(ascii_encoding)
    if remove_numb:
        df[rows_apply] = df[rows_apply].applymap(remove_number)
    if remove_underline:
        df[rows_apply] = df[rows_apply].applymap(remove_underlines)
    if trim:
        df[rows_apply] = df[rows_apply].applymap(apply_strip)
    if remove_punct:
        df[rows_apply] = df[rows_apply].applymap(remove_punctuation)
    if remove_stp:
        df[rows_apply] = df[rows_apply].applymap(remove_stopwords)
    if truncate:
        df[rows_apply] = df[rows_apply].applymap(trucate_text)
    return df


def dict_clean_given_path(path):
    clean_dict = {}
    with open(path) as file:
        for line in file:
            key, value = line.strip().split("\t")
            clean_dict[key] = value

    return clean_dict


def dict_clean_rel_ent(path_rel, path_ent):
    clean_dict_rel_ent = dict_clean_given_path(path_rel)
    clean_dict_rel_ent.update(dict_clean_given_path(path_ent))
    return clean_dict_rel_ent


def clean_data_replace_file(df, path_rel, path_ent, rows_apply=['subject', 'property', 'object']):
    clean_dict_rel_ent = dict_clean_rel_ent(path_rel, path_ent)
    df_copied = df.copy()
    df_copied[rows_apply] = df_copied[rows_apply].applymap(
        lambda x: clean_dict_rel_ent[x] if x in clean_dict_rel_ent else print("not in the key:{}".format(x)))
    return df_copied


def split_data_based_on_one_column(data, column_name):
    DICT_COL = {}

    for df_t in tqdm.tqdm(data.groupby([column_name])):
        dfs = df_t[1]
        DICT_COL[dfs[column_name].values[0]] = dfs
    return DICT_COL


def split_data_based_on_two_columns(data, column1Name, column2Name):
    DICT_COL1_COL2 = {}

    def add_col1_col2_DICT(df_t):
        dt = df_t.copy()
        pp = dt[column1Name].values[0]
        oo = dt[column2Name].values[0]
        if pp not in DICT_COL1_COL2:
            DICT_COL1_COL2[pp] = {}
        DICT_COL1_COL2[pp][oo] = dt

    for df_t in tqdm.tqdm(data.groupby([column1Name, column2Name])):
        add_col1_col2_DICT(df_t[1])

    return DICT_COL1_COL2


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
    dict_correlation_prop_obj = correlation_prop_obj(map_ent_prop_to_df, ent_pos)
    possible_ent_false_neg_dict = {}
    for ent in tqdm.tqdm(map_ent_prop_to_df.keys()):
        possible_ent_false_neg_dict[ent] = {}
        for prop in map_ent_prop_to_df[ent].keys():
            res = set(map_prop_to_df[prop][ent_pos].values)

            limit_from_this = set(map_ent_prop_to_df[ent][prop][ent_pos].values)
            tot_lim = set()
            for ee in limit_from_this:
                tot_lim |= dict_correlation_prop_obj[prop][ee]
            res -= tot_lim

            possible_ent_false_neg_dict[ent][prop] = list(res)
    return possible_ent_false_neg_dict


def raw_to_transformer_triple(triple_sentence_list):
    sentence_transformer_triple = []
    for triple in tqdm.tqdm(triple_sentence_list):
        sentence_transformer_triple.append(InputExample(texts=triple))
    return sentence_transformer_triple


def trim_prop_till_unique(_set_):
    # __set_ : unique props of data. for FB15K dataset
    old_prop_to_new = {}
    hop = {}  # akhari: kol
    i = 1
    while len(_set_) != 0:
        for prop in _set_:
            akhari = "/".join(prop.split("/")[-i:])
            if akhari not in hop:
                hop[akhari] = set([prop])
            else:
                hop[akhari].add(prop)

        for k, v in hop.items():
            if len(v) == 1:
                old_prop_to_new[list(v)[0]] = k.replace('.', ' ').replace('-', ' ').replace('–', ' ').replace('/', ' ')
                _set_ -= v
        i += 1

    props = list(old_prop_to_new.values())
    final_dict = {}
    i = 0
    while i < 10:
        i += 1
        keep_track_pp = []
        for pp in props:
            possible_p = ' '.join(pp.split(' ')[0:i])

            keep_track_pp.append(possible_p)
        rmv_pp = []
        for pp in props:
            splitted_pp = pp.split(' ')
            if len(splitted_pp) == 1:
                continue

            splitted_pp_str = ' '.join(splitted_pp[0:i])
            if keep_track_pp.count(splitted_pp_str) == 1:
                final_dict[pp] = splitted_pp_str
                rmv_pp.append(pp)

        for q in rmv_pp:
            props.remove(q)

    for k, v in old_prop_to_new.items():
        if v in final_dict:
            old_prop_to_new[k] = final_dict[v]

    return old_prop_to_new


def mapper_index_dict(dict_embeds):
    mapper_idx = {}
    i = 0
    for ent, val in dict_embeds.items():
        for idx in range(i, i + len(val)):
            mapper_idx[idx] = ent
        i += len(val)
    return mapper_idx


