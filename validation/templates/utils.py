import numpy as np
from collections import defaultdict

def nested_dict(levels):
    """.
    PAPER LINK: Section 4 (Algorithmic Logic and States) => "...we often assign descriptive variable names to improve clarity. These identifiers serve only as labels and do not affect computation."
    => "...the input structure is organized into blocks corresponding to the bits of p and q, along with their associated carry bits."
    """
    if levels == 1:
        return defaultdict(dict)
    return defaultdict(lambda: nested_dict(levels - 1))

def flatten_blocks(blocks):
    flatted_result = []
    for block_name, block in blocks.items():
        if isinstance(block, list):
            flatted_result.extend(block)
        else:
            flatted_result.append(block)
    return flatted_result

def match_template(xhat, templates):
    matches = []
    for i, template in enumerate(templates):
        match = True 
        for k1 in template:
            for k2 in template[k1]:
                if template[k1][k2] != xhat[k1][k2]:
                    match = False
                    break
            if not match:
                break
        if match:
            matches.append(i)
    return matches

def encode_data(xhat, templates):
    x = np.zeros(len(templates))
    matches = match_template(xhat, templates)
    x[matches] = 1
    if x.sum() > 0:
        x = x/np.sqrt(x.sum())
    return x