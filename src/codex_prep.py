from .prepare_data import clean_data
import pandas as pd
from codex.codex import Codex


def codex_data_stat():
    sizes = Codex.SIZES
    codes = Codex.CODES
    for size in sizes:
        codex = Codex(size=size)

        train, valid, test = [
            codex.split(split) for split in ("train", "valid", "test")]
        triples = codex.triples()

        print(codex.name())

        print(
            " ", len(codex.entities()), "entities /",
            len(codex.relations()), "relations"
        )

        print(
            " ", len(train), "train /",
            len(valid), "validation /",
            len(test), "test"
        )
        print(" ", len(triples), "total triples")


def replace_id_with_label(row, codex):
    h, r, t = tuple(row)
    row[0] = codex.entity_label(h)
    row[1] = codex.relation_label(r)
    row[2] = codex.entity_label(t)


def clean_df_replace(df, codex):
    df_cop = df.copy()
    df_cop.apply(replace_id_with_label, codex=codex, axis=1)
    df_cop = df_cop.rename(columns={"head": "subject", "relation": "property", "tail": "object"})
    return clean_data(df_cop, remove_punct=False, remove_numb=False, remove_stp=False, truncate=False)


def prepare_full_test_dev(test_df, test_neg_df):
    test_df_cc = test_df.copy()
    test_neg_df_cc = test_neg_df.copy()
    test_df_cc["y"] = "1"
    test_neg_df_cc["y"] = "-1"
    full_test = pd.concat([test_df_cc, test_neg_df_cc], ignore_index=True)
    return full_test


def add_type_df(codex):
    ent_to_type = {}
    row_lists = []

    for i, eid in enumerate(codex.entities()):
        types = codex.entity_types(eid)
        types_l = [codex.entity_type_label(etype) for etype in types]
        ent_to_type[codex.entity_label(eid)] = types_l

    for key, vals in ent_to_type.items():
        for val in vals:
            row_lists.append({'subject': key, "property": "type", "object": val})
    ent_to_type_df = pd.DataFrame(row_lists, columns=['subject', 'property', 'object'])
    # clean_data(ent_to_type_df)
    return ent_to_type_df


def add_abstract_df(codex):
    row_lists = []

    for i, eid in enumerate(codex.entities()):

        # print(f"From {codex.entity_wikipedia_url(eid)}:")
        abstract = codex.entity_extract(eid)
        if abstract == "":
            continue
        else:
            row_lists.append({'subject': codex.entity_label(eid), "property": "has abstract", "object": abstract})
    ent_to_abstract_df = pd.DataFrame(row_lists, columns=['subject', 'property', 'object'])
    # return clean_data(ent_to_abstract_df)
    return ent_to_abstract_df
