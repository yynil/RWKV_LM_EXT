import abc
import torch
class LogitsProcessor:
    def __init__(self):
        pass

    @abc.abstractmethod
    def process(self,input_ids: torch.LongTensor, logits :torch.FloatTensor) -> torch.FloatTensor:
        pass

class TopPLogitsProcess(LogitsProcessor):
    def __init__(self,top_p:float,min_tokens_to_keep:int=1):
        self.top_p = top_p
        self.min_tokens_to_keep = min_tokens_to_keep
    def process(self,input_ids: torch.LongTensor, logits :torch.FloatTensor) -> torch.FloatTensor:
        scores = torch.nn.functional.softmax(logits,dim=-1)
        sorted_scores, sorted_indices = torch.sort(scores, descending=True)
        cumulative_probs = torch.cumsum(sorted_scores, dim=-1)
        sorted_indices_to_remove = cumulative_probs > self.top_p
        size_to_remove = torch.sum(sorted_indices_to_remove, dim=-1)
        #set other than top_p to -inf
        sorted_indices_to_remove[0:self.min_tokens_to_keep] = False
        logits[sorted_indices[sorted_indices_to_remove]] = float('-inf')
        return logits
    
class TopKLogitsProcess(LogitsProcessor):
    def __init__(self,top_k:int):
        self.top_k = top_k

    def process(self,input_ids: torch.LongTensor, logits :torch.FloatTensor) -> torch.FloatTensor:
        top_k_indices = torch.topk(logits, self.top_k).indices
        all_indices = torch.arange(logits.size(-1),device=logits.device)
        mask = ~torch.isin(all_indices,top_k_indices)
        logits[mask]=float('-inf')
        return logits
    
class RepetitionPenaltyLogitsProcessor(LogitsProcessor):
    def __init__(self, penalty: float):
        if not isinstance(penalty, float) or not (penalty > 0):
            raise ValueError(f"`penalty` has to be a strictly positive float, but is {penalty}")

        self.penalty = penalty

    def process(self,input_ids: torch.LongTensor, logits :torch.FloatTensor) -> torch.FloatTensor:  
        score = torch.gather(logits, 0, input_ids)
        score = torch.where(score < 0, score * self.penalty, score / self.penalty)
        scores_processed = logits.scatter(0, input_ids, score)
        return scores_processed