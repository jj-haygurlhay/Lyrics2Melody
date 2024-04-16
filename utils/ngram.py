import collections
from collections import defaultdict
def _get_ngrams(token_list, n):
        return zip(*[token_list[i:] for i in range(n)])

def count_ngrams(token_list, n):
     return collections.Counter(x for x in _get_ngrams(token_list,n))

def ngrams_iterator(token_list, ngrams):
    """MODIFIED VERSION OF torchtext.data.utils

    Modified to work with our data.
    
    Return an iterator that yields the given tokens and their ngrams.

    Args:
        token_list: A list of tokens
        ngrams: the number of ngrams.

    Examples:
        >>> token_list = ['here', 'we', 'are']
        >>> list(ngrams_iterator(token_list, 2))
        >>> ['here', 'here we', 'we', 'we are', 'are']
    """

    for x in token_list:
        yield (x,)
    for n in range(2, ngrams + 1):
        for x in _get_ngrams(token_list, n):
            yield tuple(x)

def _compute_ngram_counter(tokens, max_n):
    """MODIFIED VERSION OF torchtext.data.metrics

    Modified to work with our data.
    
    Create a Counter with a count of unique n-grams in the tokens list

    Args:
        tokens: a list of tokens (typically a string split on whitespaces)
        max_n: the maximum order of n-gram wanted

    Outputs:
        output: a collections.Counter object with the unique n-grams and their
            associated count

    Examples:
        >>> from torchtext.data.metrics import _compute_ngram_counter
        >>> tokens = ['me', 'me', 'you']
        >>> _compute_ngram_counter(tokens, 2)
            Counter({('me',): 2,
             ('you',): 1,
             ('me', 'me'): 1,
             ('me', 'you'): 1})
    """
    assert max_n > 0
    ngrams_counter = collections.Counter(x for x in ngrams_iterator(tokens, max_n))

    return ngrams_counter

def ngram_repetition(token_list,n):
    return sum([x - 1 for x in count_ngrams(token_list,n).values()])

def transitions(notes):
    transitions = defaultdict(int)
    for bigram, count in count_ngrams(notes, 2).items():
        # transition = bigram[1] - bigram[0]
        # if transitions in transitions.keys(): transitions[transition] += count
        # else: transitions[transition] = count
        transitions[bigram[1] - bigram[0]] += count
    return transitions

def average_transitions(transitions_maps):
    average_transitions = defaultdict(int)
    for transitions_map in transitions_maps:
        for transition, count in transitions_map.items():
            if transition in average_transitions.keys(): average_transitions[transition] += count / len(transitions_maps)
            else: average_transitions[transition] = count / len(transitions_maps)
    return average_transitions


def test():
    pred = [[(1,1),(2,2),(3,3),(4,4),(1,1),(5,5)]]
    print(count_ngrams(pred[0],2))
    print(ngram_repetition(pred[0],2))
    pred = [[1,2,3,4,1,5]]
    print(count_ngrams(pred[0],2))

if __name__ =="__main__":
    test()

class transition_map (defaultdict):
    def __init__(self, notes):
        super().__init__(int)
        for bigram, count in count_ngrams(notes, 2).items():
            # transition = bigram[1] - bigram[0]
            # if transitions in transitions.keys(): transitions[transition] += count
            # else: transitions[transition] = count
            self[bigram[1] - bigram[0]] += count

    def __add__(self, other_transition_map):
        for transition, count in other_transition_map.items():
            self[transition] += count
        return self
    
    def __truediv__(self, n: int):
        for transition, count in self.items():
            self[transition] = count/n
        return self
    
    def average(transition_maps):
        transition_sum = transition_maps[0]
        for transition_map in transition_maps[1:]:
            transition_sum += transition_map
        return transition_sum / len(transition_maps)