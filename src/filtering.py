
def exist_in_tr_ts_dv(s, p, o, map_subj_prop_to_df, map_subj_prop_to_df_dev, map_subj_prop_to_df_test):

    try:
        s_o_dev = set(map_subj_prop_to_df_dev[s][p]['object'])
    except:
        s_o_dev = set()

    try:
        s_o_train = set(map_subj_prop_to_df[s][p]['object'])
    except:
        s_o_train = set()

    try:
        s_o_test = set(map_subj_prop_to_df_test[s][p]['object'])
    except:
        s_o_test = set()

    s_o = s_o_test | s_o_train | s_o_dev

    if o in s_o:
        return True
    return False


def apply_filter_on_pred_props(row, map_subj_prop_to_df, map_subj_prop_to_df_dev, map_subj_prop_to_df_test):
    subj = row.subject
    obj = row.object
    correct_p = row.property
    pred_p_l = row.closest_props_ordered

    already_in_tr_dev_test = []
    try:
        set_props = set(pred_p_l)
    except:
        print(row)
        set_props = set()
    for prop_cand in set_props:
        if prop_cand == correct_p:
            continue
        if exist_in_tr_ts_dv(subj, prop_cand, obj, map_subj_prop_to_df, map_subj_prop_to_df_dev, map_subj_prop_to_df_test):
            already_in_tr_dev_test.append(prop_cand)

    # ATTENTION: FILTERING may reduce the nb of possibile prop cands change k to higher number

    pred_p_l_f = []
    for x in pred_p_l:
        if x not in already_in_tr_dev_test:
            pred_p_l_f.append(x)

    return pred_p_l_f
