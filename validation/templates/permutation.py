import numpy as np
from .utils import nested_dict, flatten_blocks

def init_blocks(bit_length):
    # Khởi tạo block với giá trị 0 cho tất cả các vị trí
    blocks = {
        'p': [0] * bit_length
    }
    # nested_dict(2) cho phép truy cập theo kiểu x['block'][i] mà không sợ lỗi Key Error cho template
    return nested_dict(2), blocks

def init_sample(bit_length):
    return init_blocks(bit_length)

def unflatten_blocks(flat, bit_length):
    blocks = {
        'p': flat
    }
    return blocks

def get_permutation_dataset(bit_length, pi):
    """
    TẠO INPUT (TEMPLATES) và OUTPUT (Y) CHO PHÉP HOÁN VỊ
    
    Dựa trên logic của phép hoán vị: bit tại vị trí i sẽ được chuyển đến vị trí pi[i]
    """
    templates = []
    Y = []

    for i in range(bit_length):
        # Template: bit tại vị trí i được đặt là 1
        template, y = init_blocks(bit_length)
        template['p'][i] = 1
        # Output: bit tại vị trí pi[i] được đặt là 1
        y['p'][pi[i]] = 1
        templates.append(template)
        Y.append(flatten_blocks(y))
    
    return templates, np.array(Y), pi