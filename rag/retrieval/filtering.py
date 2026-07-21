def filter_indices(chunks, source=None):
    indices = set()
    for idx, chunk in enumerate(chunks):
        if source is None or chunk.source == source:
            indices.add(idx)
    return indices