import problem_unittests as tests


def create_lookup_tables(text):
    """
    Create lookup tables for vocabulary
    :param text: The text of tv scripts split into words
    :return: A tuple of dicts (vocab_to_int, int_to_vocab)
    """
    vocab_to_int = {}
    int_to_vocab = {}

    words = set(text)

    for index, word in enumerate(words):
        vocab_to_int[word] = index
        int_to_vocab[index] = word

    # return tuple
    return vocab_to_int, int_to_vocab


tests.test_create_lookup_tables(create_lookup_tables)