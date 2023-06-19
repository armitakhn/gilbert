import numpy as np
import tqdm


def ensurelistsize_ind(l, n):
    if (len(l) < n):
        l = l + [0] * (n - len(l))
    else:
        l = l[:n]
    return l


def padding_mirror(x, sentence_shape):
    number_to_add = sentence_shape - len(x)
    if number_to_add < 0:
        return x[:sentence_shape]
    each_side = int((number_to_add / 2) + 0.5)
    if number_to_add % 2 != 0:
        left = each_side - 1
        right = each_side
    else:
        left = each_side
        right = each_side

    final_x = np.pad(x, ((left, right)), 'reflect')
    return final_x


def sentencepiece_label_padding_mirror(label, embed_size, sp):
    text = sp.EncodeAsIds(label)
    return np.array(padding_mirror(text, embed_size))


def sentencepiece_label_padding_zeros(label, embed_size, sp):
    text = sp.EncodeAsIds(label)
    return np.array(ensurelistsize_ind(text, embed_size))


def encode_sentencepiece(partialFacts, sp_model, embed_size=64, padding_type='mirror'):
    if padding_type == 'mirror':
        padding = sentencepiece_label_padding_mirror

    elif padding_type == 'zeros':
        padding = sentencepiece_label_padding_zeros
    else:
        raise NotImplementedError('we only have mirror and zeros padding defined. You need to define yours here!')

    partialFact_transformed = []
    for triple_fact in tqdm.tqdm(partialFacts):
        anc = padding(triple_fact[0], embed_size, sp_model)
        pos = padding(triple_fact[1], embed_size, sp_model)
        neg = padding(triple_fact[2], embed_size, sp_model)

        partialFact_transformed.append([anc, pos, neg])

    return partialFact_transformed


def encode_sent(sentence, sp_model, embed_size=64, padding_type='mirror'):
    if padding_type == 'mirror':
        padding = sentencepiece_label_padding_mirror

    elif padding_type == 'zeros':
        padding = sentencepiece_label_padding_zeros
    else:
        raise NotImplementedError('we only have mirror and zeros padding defined. You need to define yours here!')

    return padding(sentence, embed_size, sp_model)
