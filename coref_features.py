import numpy as np
import collections
from collections import defaultdict
import torch
from utils import one_hot

FEATURE_VALUE_DICT = {
    'GENDER' : ['FEMALE', 'MALE', 'NEUTRAL', 'UNKNOWN', 'NONE'],
    'ANIMACY' : ['ANIMATE', 'INANIMATE', 'UNKNOWN', 'NONE'],
    'MENTION_TYPE' : ['LIST', 'NOMINAL', 'PRONOMINAL', 'PROPER', 'NONE'],
    'NUMBER' : ['PLURAL', 'SINGULAR', 'UNKNOWN', 'NONE']
}
FEATURE_DIM = sum(len(v) for v in FEATURE_VALUE_DICT.values())


def get_origin_indices(span, metadata):
    if (span[0] not in metadata['rev_start_maps'].keys()) or (span[1] not in metadata['rev_end_maps'].keys()):
        return None
    else:
        return (metadata['rev_start_maps'][span[0]], metadata['rev_end_maps'][span[1]])



def is_mention(spans, mention_features):
    return (spans[0], spans[1]) in mention_features.keys()

def get_mention_features(batched_metadata, indices):
    batch_feature_vecs = []
    for spans, metadata in zip(indices.data.tolist(), batched_metadata):
        feature_vecs = []
        print(metadata.keys())
        all_features = metadata['features']
        #all_features = metadata
        mention_features = all_features['mentions']

        for span in spans:
            mention_feature_dict = {'GENDER':       'NONE',
                                    'ANIMACY':      'NONE',
                                    'MENTION_TYPE': 'NONE',
                                    'NUMBER':       'NONE'}

            origin_spans = get_origin_indices(span, metadata)
            if origin_spans is not None:
                if is_mention(origin_spans, mention_features):
                    m_feature = mention_features[(origin_spans[0], origin_spans[1])]
                    mention_feature_dict['GENDER'] = m_feature.gender
                    mention_feature_dict['ANIMACY'] = m_feature.animacy
                    mention_feature_dict['MENTION_TYPE'] = m_feature.mentionType
                    mention_feature_dict['NUMBER'] = m_feature.number


            mention_feature_onehot_dict = {}
            for k in mention_feature_dict.keys():
                mention_feature_onehot_dict[k] = one_hot(FEATURE_VALUE_DICT[k].index(mention_feature_dict[k]), len(FEATURE_VALUE_DICT[k]))

            all_feature_vec = np.concatenate([mention_feature_onehot_dict['GENDER'], mention_feature_onehot_dict['ANIMACY'], mention_feature_onehot_dict['MENTION_TYPE'], mention_feature_onehot_dict['NUMBER']])
            all_feature_vec = np.expand_dims(all_feature_vec, 0)
            feature_vecs.append(all_feature_vec)
        feature_vecs = np.concatenate(feature_vecs)
        feature_vecs = np.expand_dims(feature_vecs, 0)
        batch_feature_vecs.append(feature_vecs)
    batch_feature_vecs = np.concatenate(batch_feature_vecs)
    batch_feature_vecs = batch_feature_vecs.astype(np.float32)
    batch_feature_vecs = torch.tensor(batch_feature_vecs).cuda()
    return batch_feature_vecs


