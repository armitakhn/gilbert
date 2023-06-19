from assets import TextualSequence
import faiss
import numpy as np
from tqdm import tqdm
from sent_pi import encode_sent


def create_sents_not_sb(subject, prop, obj, mode, dict_like, sp_model):
    sentence = ''
    if mode == 'so':
        sentence = f'{subject} {obj}'
    if mode == 'all':
        sentence = f'{subject} {prop} {obj}'

    encoded_sent = encode_sent(sentence, sp_model, embed_size=256)
    return encoded_sent


def create_sents_sb(subject, prop, obj, mode, dict_like, sp_model):
    textualsequence = TextualSequence(dict_like=dict_like, ent_abstract=False)

    if mode == "all":
        if not dict_like:
            return textualsequence.create_textual_sequence(subject, prop, obj)[0]
        else:
            return {'sent1': subject, 'sent2': prop, 'sent3': obj}

    if mode == "so":
        if not dict_like:
            return textualsequence.create_textual_sequence(subject, obj)[0]
        else:
            return {'sent1': subject, 'sent2': obj}

    if mode == "po":
        if not dict_like:
            return textualsequence.create_textual_sequence(prop, obj)[0]
        else:
            return {'sent1': prop, 'sent2': obj}

    if mode == "sp":
        if not dict_like:
            return textualsequence.create_textual_sequence(subject, prop)[0]
        else:
            return {'sent1': subject, 'sent2': prop}


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def find_embedding_based_on_mode(data, model, mode="all", dict_like=False, sentencebert=True, sp=None):
    ## replaces funcions such as :  find_embedding_based_on_subject_object

    if sentencebert:
        create_sents = create_sents_sb
    else:
        create_sents = create_sents_not_sb

    all_sentences = []
    TDC = data.copy()
    import torch
    from torch import Tensor
    from torch.autograd import Variable
    for idx, row in TDC.iterrows():
        sentence = create_sents(row.subject, row.property, row.object, mode, dict_like, sp)
        all_sentences.append(sentence)

    if sentencebert:
        emb_sents = model.encode(all_sentences, show_progress_bar=False)
        emb_sents = list(emb_sents)
    else:

        gen = chunks(all_sentences, 5000)
        emb_sents = []
        for l in gen:

            all_sentences = np.array(l)
            row = Variable(Tensor(np.array(all_sentences)).float())

            device = torch.device("cuda")
            rr = model(row.to(device))
            rr = rr.cpu().detach().numpy()
            emb_sents.extend(list(rr))

    TDC['embedding'] = emb_sents

    return TDC


def find_embedding_based_on_hub(data, model, mode="all", hub="subject", dict_like=False, sentencebert=True, sp=None):
    # hub can be subject or object
    df_with_embs = find_embedding_based_on_mode(data, model, mode=mode, dict_like=dict_like, sentencebert=sentencebert,
                                                sp=sp)
    dict_ent_to_embs = df_with_embs.groupby(hub)["embedding"].apply(list).apply(np.array).to_dict()
    return dict_ent_to_embs


def search_in_faiss_pipeline(test_data_embbedded, ent_stacks, k, use_gpu=True, verbose=False):
    dim = len(test_data_embbedded.iloc[0].embedding)
    if use_gpu:
        import faiss
        res = faiss.StandardGpuResources()
        index = faiss.IndexFlatL2(dim)  # build the index
        index = faiss.index_cpu_to_gpu(res, 0, index)
    else:
        import faiss

        index = faiss.IndexFlatL2(dim)  # build the index

    index.add(ent_stacks)

    if verbose:
        print("dim is: ", dim)
        print("indexing done?", index.is_trained)
        print("index.ntotal:", index.ntotal)

    # D, I = index.search(ent_stacks[:5], k) # sanity check
    # print(I)
    # print(D)

    xq = np.vstack(test_data_embbedded.embedding[0:])

    D, I = index.search(xq, k)  # actual search
    # print(I)                   # neighbors of the 5 first queries
    # print(I[-5:])                  # neighbors of the 5 last queries
    # print("I has shape:", I.shape)
    return D, I


def add_closest_properties(I, mapper_idx_prop):
    closest_p = []
    for i in I:
        mapped_to_prop_i = [mapper_idx_prop[j] for j in i if j != -1]
        # key error on -1 TODO: check if the if clause is ok
        # [11  8 10 16 17  5 18 15 13  2 19  0 12  1  3  4 14  9  6  7 -1 -1 -1 -1
        # -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
        # -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1

        closest_p.append(mapped_to_prop_i)
    return closest_p
