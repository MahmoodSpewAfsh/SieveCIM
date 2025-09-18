#!/usr/bin/env python3
import numpy as np
import itertools

# ----------------------------
# Your original build_meta_with_fixed (single authoritative copy)
# ----------------------------
def build_meta_with_fixed3(B: np.ndarray, bits: int = 3, fixed_idx: int = None, fixed_val: int = 1, convention: str = "rows"):
    """
    Build meta dictionary describing binary encoding of lattice coefficients u.
    - B: integer basis (rows or columns, see convention)
    - bits: bits per non-fixed coefficient (last bit is sign)
    - fixed_idx/fixed_val: fix certain coefficient indices to given integer(s)
    - convention: "rows" (use u @ B) or "columns" (use B @ u)
    Returns meta dict with keys used by the rest of the script.
    """
    B = np.asarray(B, dtype=np.int64)
    if convention == "rows":
        n = B.shape[0]
        G = (B @ B.T).astype(np.int64)
    elif convention == "columns":
        n = B.shape[1]
        G = (B.T @ B).astype(np.int64)
    else:
        raise ValueError("convention must be 'rows' or 'columns'")

    if fixed_idx is None:
        fixed_indices = []
        fixed_vals = {}
    else:
        if isinstance(fixed_idx, int):
            fixed_indices = [fixed_idx]
            fixed_vals = {fixed_idx: int(fixed_val)}
        else:
            fixed_indices = list(fixed_idx)
            if hasattr(fixed_val, '__iter__') and not isinstance(fixed_val, (str, bytes)):
                fixed_vals = {i: int(v) for i, v in zip(fixed_indices, fixed_val)}
            else:
                fixed_vals = {i: int(fixed_val) for i in fixed_indices}

    var_idx = 0
    var_index_map = {}
    bit_weights = {}
    for i in range(n):
        if i in fixed_indices:
            continue
        for k in range(bits - 1):
            var_index_map[(i, k)] = var_idx
            bit_weights[(i, k)] = 1 << k
            var_idx += 1
        var_index_map[(i, bits - 1)] = var_idx
        # sign bit (negative weight)
        bit_weights[(i, bits - 1)] = -(1 << (bits - 1))
        var_idx += 1

    m = var_idx
    return {
        'n': n,
        'bits': bits,
        'var_index_map': var_index_map,
        'bit_weights': bit_weights,
        'm': m,
        'G': G,
        'convention': convention,
        'fixed_indices': fixed_indices,
        'fixed_vals': fixed_vals,
        'B': B,
    }

# ----------------------------
# Decode, energy helpers (same semantics as before)
# ----------------------------
def decode_bits_to_u_with_fixed(x_bits: np.ndarray, meta: dict) -> np.ndarray:
    n = meta['n']
    bits = meta['bits']
    var_map = meta['var_index_map']
    bit_weights = meta['bit_weights']
    u = np.zeros(n, dtype=np.int64)
    for i in range(n):
        if i in meta['fixed_indices']:
            u[i] = int(meta['fixed_vals'][i])
            continue
        s = 0
        for k in range(bits):
            vi = var_map[(i, k)]
            s += int(x_bits[vi]) * int(bit_weights[(i, k)])
        u[i] = int(s)
    return u

def energy_from_x_bits(x_bits: np.ndarray, meta: dict):
    """
    Compute lattice energy E = u^T G u and return u and v.
    """
    u = decode_bits_to_u_with_fixed(x_bits, meta)
    G = meta['G']
    B = meta['B']
    if meta['convention'] == 'rows':
        v = (u @ B).astype(np.int64)
    else:
        v = (B @ u).astype(np.int64)
    e = int(u @ (G @ u))
    return e, u, v

# ----------------------------
# Build full QUBO (Q, lin, const)
# ----------------------------
def build_qubo_full(meta: dict):
    """
    Returns Q (m x m), lin (m,), const scalar such that
       E(x) = x^T Q x + lin^T x + const = u(x)^T G u(x)
    """
    m = meta['m']
    Q = np.zeros((m, m), dtype=np.int64)
    lin = np.zeros(m, dtype=np.int64)
    const = 0

    G = meta['G']
    var_map = meta['var_index_map']
    bit_weights = meta['bit_weights']
    fixed_idx = set(meta['fixed_indices'])
    fixed_vals = meta['fixed_vals']
    n = meta['n']

    # u_fixed
    u_fixed = np.zeros(n, dtype=np.int64)
    for i in range(n):
        if i in fixed_idx:
            u_fixed[i] = int(fixed_vals[i])

    # Quadratic free-free
    for (i, ki), vi in var_map.items():
        wi = int(bit_weights[(i, ki)])
        for (j, kj), vj in var_map.items():
            wj = int(bit_weights[(j, kj)])
            Q[vi, vj] += wi * wj * int(G[i, j])

    # Linear from free-fixed cross terms: 2 * u_free^T G u_fixed
    for (i, ki), vi in var_map.items():
        wi = int(bit_weights[(i, ki)])
        s = 0
        for j in range(n):
            if j in fixed_idx:
                s += int(G[i, j]) * int(u_fixed[j])
        lin[vi] += 2 * wi * s

    # Constant from fixed-fixed
    const = int(u_fixed @ (G @ u_fixed))

    return Q, lin, const

# ----------------------------
# Build Q_single and brute force checker
# ----------------------------
def qubo_single_from_full(Q, lin, const):
    """
    Q_single = Q + diag(lin)
    Energy: E(x) = x^T Q_single x + const
    """
    Q_single = Q.copy().astype(np.int64)
    for i in range(Q_single.shape[0]):
        Q_single[i, i] += int(lin[i])
    return Q_single, int(const)

def brute_force_qubo_single(Q_single: np.ndarray, const: int):
    """
    Brute force min E(x) = x^T Q_single x + const over x in {0,1}^m.
    Returns (E_min, list_of_minimizers).
    """
    m = Q_single.shape[0]
    if m > 24:
        raise ValueError("Brute force will be slow for m > 24 (2^m states).")
    best_E = None
    winners = []
    for tup in itertools.product([0,1], repeat=m):
        x = np.fromiter(tup, dtype=np.int64)
        E = int(x @ (Q_single @ x)) + const
        if best_E is None or E < best_E:
            best_E = E
            winners = [x.copy()]
        elif E == best_E:
            winners.append(x.copy())
    return int(best_E), winners

# ----------------------------
# Verification routine (ties Q_single with decoded lattice energy)
# ----------------------------
def verify_Q_single(meta: dict, verbose: bool = True):
    Q, lin, const = build_qubo_full(meta)
    Q_single, c = qubo_single_from_full(Q, lin, const)
    E_q, winners = brute_force_qubo_single(Q_single, c)
    print(Q_single[:3])
    # decode winners and cross-check with decoded lattice energy
    decoded = []
    for x in winners:
        e_dec, u_dec, v_dec = energy_from_x_bits(x, meta)
        decoded.append((e_dec, x.copy(), u_dec.copy(), v_dec.copy()))

    all_ok = all(e_dec == E_q for (e_dec, x, u, v) in decoded)
    if verbose:
        print("Q_single shape:", Q_single.shape)
        print("const:", c)
        print("Q_single (first rows):\n", Q_single[:min(8, Q_single.shape[0]), :min(8, Q_single.shape[1])])
        print("\nQ_single brute-force min E:", E_q)
        print("Number of minimizers (binary x):", len(winners))
        for idx, (e_dec, x, u, v) in enumerate(decoded):
            print(f" Minimizer #{idx+1}: E_dec={e_dec}, x={x.tolist()}, u={u.tolist()}, v={v.tolist()}, ||v||^2={int(v @ v)}")
        print("\nVerification all match? ->", all_ok)
    return {
        'Q_single': Q_single,
        'const': c,
        'E_qubo_min': E_q,
        'winners_x': winners,
        'decoded': decoded,
        'verified': all_ok
    }

def primal(A, b):
    M = np.zeros((2 * int(len(A[0])) + 1, 2 * int(len(A[0])) + 1), dtype= np.short)

    for i in range(int(len(A[0]))):
        M[i, i] = 1

    for i in range(int(len(A[0]))):
        for j in range(int(len(A[0])), 2 * int(len(A[0]))):
            M[i, j] = - int(A[i, j - int(len(A[0]))])

    for i in range(int(len(A[0])), 2 * int(len(A[0]))):
        M[i, i] = 3329

    for i in range(int(len(A[0])), 2 * int(len(A[0]))):
        M[2 * int(len(A[0])), i] = int(b[i - int(len(A[0]))])

    M[2 * int(len(A[0])), 2 * int(len(A[0]))] = 1

    # M = M.tolist()
    return M

# ----------------------------
# Demo usage (uses the single build_meta_with_fixed)
# ----------------------------
if __name__ == "__main__":
    # Demo basis (4x4) similar to your last example
    B_demo = np.array([[2,1,0,5],
                       [0,1,1,0],
                       [0,0,1,-5],
                       [1,2,1,-5]], dtype=np.int64)
    s = np.load("Instances/s3.npy")
    print(s)

    A = np.load("Instances/A3.npy")
    print(A)

    b = np.load("Instances/b3.npy")
    print(b)

    Bai = primal(A, b)
    print(Bai)
    B_demo = Bai 


    bits = 3
    # fixed_idx = 6
    # fixed_val = 1
    # bits = 4
    meta_demo = build_meta_with_fixed3(B_demo, bits=bits, fixed_idx=6, fixed_val=1, convention="rows")
    out = verify_Q_single(meta_demo, verbose=True)
