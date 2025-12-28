import numpy as np
from .utils import nested_dict, flatten_blocks

def init_blocks(bit_length):
    # Khởi tạo block với giá trị 0 cho tất cả các vị trí
    blocks = {
        'sum_p': [0] * bit_length,
        'sum_q': [0] * bit_length,
        'sum_c': [0] * bit_length
    }
    # nested_dict(2) cho phép truy cập theo kiểu x['block'][i] mà không sợ lỗi Key Error cho template
    return nested_dict(2), blocks

def unflatten_blocks(flat, bit_length):
    blocks = {
        'sum_p': flat[:bit_length],
        'sum_q': flat[bit_length : 2 * bit_length],
        'sum_c': flat[2 * bit_length : 3 * bit_length],
    }
    return blocks

def get_addition_dataset(bit_length):
    """
    TẠO INPUT (TEMPLATES) và OUTPUT (Y) CHO PHÉP CỘNG NHỊ PHÂN
    
    PAPER:
    - Section 5.1: Xây dựng tập hợp các Template-label tuples (instructions).
    - Appendix B.2: Các quy tắc logic cho bộ cộng Ripple-Carry.
    - Ví dụ cụ thể của nó chính là Figure 2
      + Nhma vector y trong cài đặt lại khác với Figure 2 một chút.
      + Trong Figure 2 author flats xen kẽ q1 p1 c1 q2 p2 c2, nhma với implement thì ta dùng hàm flatten_blocks sẽ cho ra q1 q2 p1 p2 c1 c2 
      => Vị trí của các bit trong vector thực chất chỉ là những 'địa chỉ' cố định; chỉ cần giữ nguyên quy ước sắp xếp (Flatten) xuyên suốt từ lúc huấn luyện đến lúc thực tế, AI sẽ tự học được mối liên kết logic giữa các địa chỉ đó, và ta sẽ dùng hàm unflatten cắt đúng các đoạn đã hứa chính là cách để thu về kết quả chính xác.
    """
    templates = []
    Y = []

    for i in range(bit_length):
        # Implement các phần templates có trong B.2
        # Templates: Bitwise addition
        template, y = init_blocks(bit_length)
        template['sum_p'][i] = 0
        template['sum_q'][i] = 1
        y['sum_p'][i] = 1
        templates.append(template)
        Y.append(flatten_blocks(y))

        # Templates: Bitwise addition
        template, y = init_blocks(bit_length)
        template['sum_p'][i] = 1
        template['sum_q'][i] = 0
        y['sum_p'][i] = 1
        templates.append(template)
        Y.append(flatten_blocks(y))

        # Templates: Carry propagation
        template, y = init_blocks(bit_length)
        template['sum_p'][i] = 1
        template['sum_q'][i] = 1
        y['sum_p'][i] = 0
        y['sum_c'][i] = 1
        templates.append(template)
        Y.append(flatten_blocks(y))

        # Templates: termination
        template, y = init_blocks(bit_length)
        template['sum_c'][i] = 1
        if i < bit_length - 1:
            y['sum_q'][i+1] = 1
        else:
            y['sum_c'][i] = 1
        templates.append(template)
        Y.append(flatten_blocks(y))
    
    return templates, np.array(Y)