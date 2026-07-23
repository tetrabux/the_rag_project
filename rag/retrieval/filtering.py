def filter_indices(chunks, source=None):
    # a set, so the `i in allowed_indices` checks in dense/lexical search are O(1)
    indices = set()
    for idx, chunk in enumerate(chunks):
        if source is None or chunk.source == source:
            indices.add(idx)
    return indices