import random
import torch
def pad_and_truncated(features, max_len, pad_token_id=0,eos_token_id=1):
    query_ids = [feature['query'] for feature in features]
    rand_pos_ids = [random.choice(p) for p in [feature['pos'] for feature in features]]
    rand_neg_ids = [random.choice(n) for n in [feature['neg'] for feature in features]]
    query_ids = [q[:max_len-1]+[eos_token_id] for q in query_ids]
    rand_pos_ids = [p[:max_len-1]+[eos_token_id] for p in rand_pos_ids]
    rand_neg_ids = [n[:max_len-1]+[eos_token_id] for n in rand_neg_ids]
    query_ids = [q+[pad_token_id]*(max_len-len(q)) for q in query_ids]
    rand_pos_ids = [p+[pad_token_id]*(max_len-len(p)) for p in rand_pos_ids]
    rand_neg_ids = [n+[pad_token_id]*(max_len-len(n)) for n in rand_neg_ids]
    return {'query':torch.tensor(query_ids,dtype=torch.long),
            'positive':torch.tensor(rand_pos_ids,dtype=torch.long),
            'negative':torch.tensor(rand_neg_ids,dtype=torch.long)}    

def pad_and_truncated_according_data(features, pad_token_id=0,eos_token_id=1):
    max_len = features['fixed_len'][0]
    query_ids = [feature['query'] for feature in features]
    rand_pos_ids = [random.choice(p) for p in [feature['pos'] for feature in features]]
    rand_neg_ids = [random.choice(n) for n in [feature['neg'] for feature in features]]
    query_ids = [q[:max_len-1]+[eos_token_id] for q in query_ids]
    rand_pos_ids = [p[:max_len-1]+[eos_token_id] for p in rand_pos_ids]
    rand_neg_ids = [n[:max_len-1]+[eos_token_id] for n in rand_neg_ids]
    query_ids = [q+[pad_token_id]*(max_len-len(q)) for q in query_ids]
    rand_pos_ids = [p+[pad_token_id]*(max_len-len(p)) for p in rand_pos_ids]
    rand_neg_ids = [n+[pad_token_id]*(max_len-len(n)) for n in rand_neg_ids]
    return {'query':torch.tensor(query_ids,dtype=torch.long),
            'positive':torch.tensor(rand_pos_ids,dtype=torch.long),
            'negative':torch.tensor(rand_neg_ids,dtype=torch.long)}    