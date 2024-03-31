import collections
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

    def _get_ngrams(n):
        return zip(*[token_list[i:] for i in range(n)])

    for x in token_list:
        yield (x,)
    for n in range(2, ngrams + 1):
        for x in _get_ngrams(n):
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