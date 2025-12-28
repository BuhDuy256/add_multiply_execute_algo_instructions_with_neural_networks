import numpy as np
import jax
import jax.numpy as jnp
import neural_tangents as nt

from neural_tangents import stax
from tqdm import tqdm

from templates.permutation import *
from templates.utils import *

max_length = 4
epsilon = 1e-10
PRINT_ALL = True   # đổi False nếu không muốn in tất cả

jax.config.update("jax_enable_x64", True)

def apply_pi(bits, pi):
    out = [0] * len(bits)
    for i, b in enumerate(bits):
        if b == 1:
            out[pi[i]] = 1
    return out

if __name__ == "__main__":
    for length in range(1, max_length + 1):
        if length == 2:
            pi = np.array([1, 0])  # swap 2 bit để nhìn rõ
        else:
            pi = np.random.permutation(length)
        X, Y_train, pi = get_permutation_dataset(length, pi=pi)
        X_train = jnp.eye(len(X))
        Y_train = jnp.array(Y_train, dtype=jnp.float64)

        out_length = len(X)

        init_fn, apply_fn, kernel_fn = stax.serial(
            stax.Dense(1024),
            stax.Relu(),
            stax.Dense(out_length)
        )

        predict_fn = nt.predict.gradient_descent_mse_ensemble(kernel_fn, X_train, Y_train)

        print(f"\n=== length={length}, pi={pi.tolist()} ===")

        for p in tqdm(range(2 ** length)):
            _, X_test = init_sample(length)
            # chuyển p thành dãy bits
            bits = np.array([int(x) for x in np.binary_repr(p, width=length)])[::-1].tolist()
            X_test['p'] = bits
            X_test_encoded = encode_data(X_test, X)

            X_test_input = jnp.array(X_test_encoded, dtype=jnp.float64).reshape(1, -1)

            y_pred = predict_fn(x_test=X_test_input, get='ntk', compute_cov=True)
            y_pred_round = np.where(y_pred.mean[0] > epsilon, 1.0, 0.0).astype(int).tolist()
            X_pred = unflatten_blocks(y_pred_round, length)

            expected = apply_pi(bits, pi.tolist())

            # --- PRINT ---
            if PRINT_ALL:
                print(f"p={p:>{len(str(2**length-1))}}  input={bits}  pred={X_pred['p']}  expected={expected}")
            else:
                # chỉ in vài case tiêu biểu
                if p in {0, 1, (2**length)-2, (2**length)-1}:
                    print(f"p={p}  input={bits}  pred={X_pred['p']}  expected={expected}")

            assert expected == X_pred['p'], (
                f"Permutation error at {p}, expected={expected}, got={X_pred['p']}"
            )

        print(f"Permutation(pi={pi.tolist()}) - bit length: {length} - all correct")