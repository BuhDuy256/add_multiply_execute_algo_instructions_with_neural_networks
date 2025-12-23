import numpy as np
import jax
import jax.numpy as jnp
import neural_tangents as nt

from neural_tangents import stax
from itertools import product
from tqdm import tqdm

from templates.addition import *
from templates.utils import *

max_length = 10 # max bits length mà ta muốn test
epsilon = 1e-10 # Biến epsilon dùng cho bước làm tròn

jax.config.update("jax_enable_x64", True) # Dùng x64 để chống việc tích tụ sai số làm tròn sau mỗi lần tính

if __name__ == "__main__":
    for length in range(1, max_length + 1):
        X, Y_train = get_addition_dataset(length)
        X_train = jnp.eye(len(X)) 
        # Trực giao hóa "địa chỉ" => Nó sẽ tạo ra các vector trực giao trong không gian ứng với địa chỉ của template đó. => Template Matching ở đây là so khớp template address trong không gian chứ không phải so khớp raw template.
        Y_train = jnp.array(Y_train, dtype=jnp.float64) # Set Target cho Neural Tangents là các Y_train đã được flatten (dtype=jnp.float64 là để tương thích với cấu hình x64)
        
        out_length = len(X)
        
        # Section 3/ Model and NTK Results
        # Docs: https://neural-tangents.readthedocs.io/en/latest/_autosummary/neural_tangents.stax.serial.html#
        init_fn, apply_fn, kernel_fn = stax.serial(
            stax.Dense(1024),
            stax.Relu(),
            stax.Dense(out_length)
        )
        
        # Section 3/ Theorem 3.1 => Khi width của network tiến tới inf thì nó sẽ hội tụ thành nt.predict.gradient_descent_mse_ensemble
        # Source code's gradient_descent_mse_ensemble: https://neural-tangents.readthedocs.io/en/latest/_modules/neural_tangents/_src/predict.html#gradient_descent_mse_ensembles
        # Trong source code của thư neural_tangents họ có nói rõ là họ references từ "Wide Neural Networks of Any Depth Evolve as "Linear Models Under Gradient Descent" của Jaehoon Lee => Cũng là paper từ tác giả của Theorem 3.1
        predict_fn = nt.predict.gradient_descent_mse_ensemble(kernel_fn, X_train, Y_train)
        
        pairs = product(list(range(2 ** length)), list(range(2 ** length)))
        
        # Sửa lỗi cú pháp :: và đảm bảo biến chạy không trùng với vòng lặp trong
        for pair_idx in tqdm(range(2 ** (2 * length))):
            p, q = next(pairs)
            out = p + q
            
            _, X_test = init_blocks(length)
            # Chuyển đổi p và q thành binary form, sau đó đảo chuỗi lại ([::-1]) đây bit thấp nhất sang trái để dễ tính
            X_test['sum_p'] = np.array([int(x) for x in np.binary_repr(p, width=length)])[::-1].tolist()
            X_test['sum_q'] = np.array([int(x) for x in np.binary_repr(q, width=length)])[::-1].tolist()

            # Section 5.1
            # Lý do "2 *" là để xứ lí cả block lan truyền bit nhớ 
            for i in range(2 * length):
                # Nó sẽ tạo ra một vector có độ dài ngang với X, sau đó duyệt qua danh sách các templates (là X) để check xem nó match với template nào, sau đó sẽ bật bit ứng tại vị trí idx của template đó (còn thực hiện chuẩn hóa x / np.sqrt(x.sum) dể vector có độ dài đơn vị (= 1))
                # Nó sẽ là một vector chứa câu trả lời cho câu hỏi: "Current state đang match với các templates nào?"
                X_test_encoded = encode_data(X_test, X)
                
                # Lí do vector này cần có độ dài đơn vị là vì trong Section 3/ Theorem 3.1 yêu cầu mọi đầu vào ||xhat|| <= 1 thì NTK mới hội tụ về phân phối Gaussian (Đọc lại Theorem 3.1) 
                X_test_input = jnp.array(X_test_encoded, dtype=jnp.float64).reshape(1, -1)
                
                # X_test lúc này sẽ là một vector chứa rất nhiều địa chỉ của các templates trong không gian trực giao => Không gian vector trực giao có đặc điểm là tích vô hướng của các vector luôn = 0 => Khi match X_test với các địa chỉ của các templates đã được trực giao hóa trước đó thì không gây xung đột kết quả trong vector y trả ra => y trả ra sẽ là một vector hợp của các y_train trước đó (không gây xung đột kết quả khi hợp) => Section 5/ Theorem 5.1 đã chứng minh
                y_pred = predict_fn(x_test=X_test_input, get='ntk', compute_cov=True)
                y_pred_round = np.where(y_pred.mean[0] > epsilon, 1.0, 0.0).tolist() # Dùng epsilon đã định nghĩa
                
                # Cập nhật X_test cho lần tính sau
                X_test = unflatten_blocks(y_pred_round, length)
            
            # Chuyển đổi kết quả out = p + q thành binary form (Sửa m thành length)
            binary_out = np.array([int(x) for x in np.binary_repr(out, width=length+1)])[::-1].tolist()
            
            # Dùng assert để kiểm tra predicted_y với kết quả thực sự
            assert binary_out[-1] == X_test['sum_c'][-1], f"Carry bit error at {p}+{q}"
            assert binary_out[:-1] == X_test['sum_p'], f"Sum bit error at {p}+{q}"

        print(f"Addition - bit length: {length} - all correct")
    print("Addition complete")