import math
import torch

from .ngram import _compute_ngram_counter
def bleu_score(candidate_corpus, references_corpus, max_n=4, weights=None):
    """MODIFIED VERSION OF torchtext.data.metrics

    Modified to work with our data.
    
    Computes the BLEU score between a candidate translation corpus and a references
    translation corpus. Based on https://www.aclweb.org/anthology/P02-1040.pdf

    Args:
        candidate_corpus: an iterable of candidate translations. Each translation is an
            iterable of tokens
        references_corpus: an iterable of iterables of reference translations. Each
            translation is an iterable of tokens
        max_n: the maximum n-gram we want to use. E.g. if max_n=3, we will use unigrams,
            bigrams and trigrams
        weights: a list of weights used for each n-gram category (uniform by default)

    Examples:
        >>> from torchtext.data.metrics import bleu_score
        >>> candidate_corpus = [['My', 'full', 'pytorch', 'test'], ['Another', 'Sentence']]
        >>> references_corpus = [[['My', 'full', 'pytorch', 'test'], ['Completely', 'Different']], [['No', 'Match']]]
        >>> bleu_score(candidate_corpus, references_corpus)
            0.8408964276313782
    """

    if weights is None:
        weights = [1/max_n]*max_n
    assert max_n == len(weights), 'Length of the "weights" list has be equal to max_n'
    assert len(candidate_corpus) == len(
        references_corpus
    ), "The length of candidate and reference corpus should be the same"

    clipped_counts = torch.zeros(max_n)
    total_counts = torch.zeros(max_n)
    weights = torch.tensor(weights)

    candidate_len = 0.0
    refs_len = 0.0

    for (candidate, refs) in zip(candidate_corpus, references_corpus):
        current_candidate_len = len(candidate)
        candidate_len += current_candidate_len

        # Get the length of the reference that's closest in length to the candidate
        if len(refs.shape) == 1:
            refs = [refs]
        refs_len_list = [float(len(ref)) for ref in refs]
        refs_len += min(refs_len_list, key=lambda x: abs(current_candidate_len - x))

        reference_counters = _compute_ngram_counter(refs[0], max_n)
        for ref in refs[1:]:
            reference_counters = reference_counters | _compute_ngram_counter(ref, max_n)

        candidate_counter = _compute_ngram_counter(candidate, max_n)

        clipped_counter = candidate_counter & reference_counters

        for ngram, count in clipped_counter.items():
            clipped_counts[len(ngram) - 1] += count

        for i in range(max_n):
            # The number of N-grams in a `candidate` of T tokens is `T - (N - 1)`
            total_counts[i] += max(current_candidate_len - i, 0)

    if min(clipped_counts) == 0:
        return 0.0
    else:
        pn = clipped_counts / total_counts
        log_pn = weights * torch.log(pn)
        score = torch.exp(sum(log_pn))

        bp = math.exp(min(1 - refs_len / candidate_len, 0))

        return bp * score.item()
    

def test():
    pred = [[(1,1),(2,2),(3,3),(4,4),(1,1),(5,5)]]
    targets = [[[(7,7),(3,3),(6,6),(2,2),(4,4),(1,1),(5,5)],[(6,6),(2,2),(3,3),(4,4),(1,1),(5,5)]]]
    assert bleu_score(pred,targets) == 0.7598356604576111
    pred = [[1,2,3,4,1,5]]
    targets = [[[7, 3,6,2,4,1,5],[6,2,3,4,1,5]]]
    assert bleu_score(pred,targets) == 0.7598356604576111
    pred = [[1,2,3,4,1,5],[1,2,3,4,1,5]]
    targets = [[[7, 3,6,2,4,1,5],[6,2,3,4,1,5]],[[7, 3,6,2,4,1,5],[6,2,3,4,1,5]]]
    print(bleu_score(pred, targets))
if __name__ == "__main__":
    test()