# Building and running the requested code: QUBO-like enumeration with the last coordinate fixed to 1.
# This matches the user's rows convention (v = u @ B, G = B @ B.T), and removes the last coordinate from variables.

import numpy as np
import itertools
from typing import Iterable, List, Tuple, Dict, Any
import matplotlib.pyplot as plt
from collections import deque
from fpylll import IntegerMatrix, SVP, LLL, BKZ
import dimod
from singlequbo import *
from IsingMachine import *
from dwave.samplers import SteepestDescentSampler

def to_numpy(x):
    """Convert tensor-like or sequence to a NumPy array without importing torch."""
    if isinstance(x, np.ndarray):
        return x
    # if it supports .cpu() (e.g. a torch tensor on GPU), try that then .numpy()
    if hasattr(x, "cpu"):
        try:
            return x.cpu().numpy()
        except Exception:
            pass
    # otherwise try .numpy()
    if hasattr(x, "numpy"):
        try:
            return x.numpy()
        except Exception:
            pass
    # fallback: try to coerce to numpy array
    try:
        return np.array(x)
    except Exception as e:
        raise TypeError("Cannot convert object to numpy array") from e

def _build_value_to_bits_lookup_for_coord(i: int, bits: int, bit_weights: Dict[tuple,int]) -> Dict[int, np.ndarray]:
    """
    For coordinate i build dict: integer_value -> bit_vector (length bits, dtype=np.int8)
    bit_weights: meta['bit_weights'] mapping (i,k) -> weight
    """
    weights = np.array([bit_weights[(i, k)] for k in range(bits)], dtype=int)  # ordered k=0..bits-1
    all_patterns = np.arange(1 << bits, dtype=np.int64)  # 0 .. 2^bits-1
    # expand bits in shape (2^bits, bits)
    bit_matrix = ((all_patterns[:, None] >> np.arange(bits)) & 1).astype(np.int8)
    # compute values for each pattern: dot with weights
    vals = (bit_matrix * weights[None, :]).sum(axis=1).astype(int)
    lookup = {}
    for idx, val in enumerate(vals):
        if val not in lookup:
            lookup[val] = bit_matrix[idx].copy()
    return lookup

def build_all_coord_lookups(meta: Dict[str, Any]) -> Dict[int, Dict[int, np.ndarray]]:
    """
    Return dict: coord_i -> { value -> bit_vector }, only for free coords.
    """
    bits = meta['bits']
    bit_weights = meta['bit_weights']
    lookups = {}
    # iterate over coordinates that have variables (i not fixed)
    free_coords = sorted({i for (i,k) in meta['var_index_map'].keys()})
    for i in free_coords:
        lookups[i] = _build_value_to_bits_lookup_for_coord(i, bits, bit_weights)
    return lookups

def v_to_u_from_B(v: np.ndarray, B: np.ndarray, convention: str='rows', tol: float = 1e-6) -> Tuple[np.ndarray, bool]:
    """
    Recover integer u from lattice vector v and basis B.
    - convention 'rows': v = u @ B  -> solve B^T u^T = v^T  (np.linalg.solve(B.T, v))
    - convention 'columns': v = B @ u -> solve B u = v
    Returns (u_int, ok_flag) where ok_flag signals whether rounding produced an exact integer solution within tol.
    """
    v = np.asarray(v, dtype=float).ravel()
    if convention == 'rows':
        # Solve B^T u^T = v^T
        sol = np.linalg.solve(B.T.astype(float), v)
    else:
        sol = np.linalg.solve(B.astype(float), v)
    u_rounded = np.rint(sol).astype(np.int64)
    # check residual
    if convention == 'rows':
        resid = B.T.astype(float) @ u_rounded - v
    else:
        resid = B.astype(float) @ u_rounded - v
    ok = float(np.linalg.norm(resid)) <= tol
    return u_rounded, ok

def u_to_bits_vector(u: np.ndarray, meta: Dict[str, Any], lookups: Dict[int, Dict[int, np.ndarray]]) -> Tuple[np.ndarray, bool]:
    """
    Convert full integer u (length n) to x_bits vector (length m) using meta.var_index_map mapping.
    Returns (x_bits, ok). ok==False if some u[i] not representable with given bits.
    """
    m = meta['m']
    bits = meta['bits']
    var_map = meta['var_index_map']   # (i,k) -> index
    fixed_indices = set(meta.get('fixed_indices', []))
    x = np.zeros(m, dtype=np.int8)
    ok = True
    # For each free coord i, find bit pattern that maps to u[i]
    for (i, k), var_idx in var_map.items():
        # we'll fill later by fetching full pattern for each i
        pass
    # Instead iterate by free coord
    free_coords = sorted({i for (i,k) in var_map.keys()})
    for i in free_coords:
        val = int(u[i])
        lookup = lookups.get(i, {})
        pattern = lookup.get(val, None)
        if pattern is None:
            # not representable
            ok = False
            break
        # pattern is array length bits with bit k at pattern[k]
        for k in range(bits):
            var_idx = var_map[(i, k)]
            x[var_idx] = int(pattern[k])
    return x, ok

def bits_to_spin_dicts(x_bits_array: np.ndarray, label_type: str = 'str') -> List[Dict[Any,int]]:
    """
    Convert X_bits array (num_samples, m) into list of dicts mapping '1'..'m' (or int 1..m if label_type='int')
    to spins in {+1,-1}. Default label type is 'str' to match your Q_dict keys like '1','2',...
    """
    x_bits_array = np.asarray(x_bits_array, dtype=np.int8)
    num_samples, m = x_bits_array.shape
    # spins s = 1 - 2*x
    S = (1 - 2 * x_bits_array).astype(int)   # +1 / -1
    dicts = []
    for r in range(num_samples):
        row = S[r]
        if label_type == 'str':
            d = {str(i+1): int(row[i]) for i in range(m)}
        else:
            d = {i+1: int(row[i]) for i in range(m)}
        dicts.append(d)
    return dicts

def gauss_vs_to_ising_states(V: np.ndarray,
                             B: np.ndarray,
                             meta: Dict[str, Any],
                             label_type: str = 'str',
                             tol_u_solve: float = 1e-6
                             ) -> Dict[str, Any]:
    """
    Main helper: convert Gauss-sieve output vectors V (shape (num_samples, n) or list) to:
       - X_bits (num_samples, m)
       - S_spins (num_samples, m)
       - ising_dicts (list of dicts mapping '1'..'m' -> spin)
       - u_array (num_samples, n) and v_array, norms_sq
    If some vector cannot be represented with given 'bits' it is flagged (valid_mask).
    """
    # normalize V to ndarray shape (num_samples, n)
    if isinstance(V, np.ndarray):
        if V.ndim == 1:
            V = V.reshape(1, -1)
        elif V.ndim != 2:
            raise ValueError("V must be 1-D or 2-D ndarray or list of vectors")
        V_arr = V.copy().astype(np.int64)
    else:
        V_arr = np.vstack([np.asarray(v, dtype=np.int64).ravel() for v in list(V)])
    num, n = V_arr.shape

    # precompute lookups per coordinate
    lookups = build_all_coord_lookups(meta)

    X_bits = np.zeros((num, meta['m']), dtype=np.int8)
    U = np.zeros((num, meta['n']), dtype=np.int64)
    norms_sq = np.zeros(num, dtype=np.int64)
    valid_mask = np.ones(num, dtype=bool)

    for idx in range(num):
        v = V_arr[idx].astype(np.int64)
        u_int, ok_solve = v_to_u_from_B(v, B, convention=meta.get('convention','rows'), tol=tol_u_solve)
        if not ok_solve:
            # mark invalid and continue
            valid_mask[idx] = False
            U[idx] = u_int
            X_bits[idx] = 0
            norms_sq[idx] = int(np.dot(v, v))
            continue
        U[idx] = u_int
        # convert u -> bits
        x_bits, ok_bits = u_to_bits_vector(u_int, meta, lookups)
        if not ok_bits:
            valid_mask[idx] = False
        X_bits[idx] = x_bits
        norms_sq[idx] = int(np.dot(v, v))

    # convert bits -> spins and dicts
    S_spins = (1 - 2 * X_bits).astype(int)   # shape (num, m)
    ising_dicts = bits_to_spin_dicts(X_bits, label_type=label_type)

    return {
        'X_bits': X_bits,
        'S_spins': S_spins,
        'ising_dicts': ising_dicts,
        'U': U,
        'V': V_arr,
        'norms_sq': norms_sq,
        'valid_mask': valid_mask,
        'lookups': lookups
    }



def spins_array_to_u_v(spins_array: np.ndarray, labels: list, meta: dict, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert spins_array (num_samples x m) -> (U, V, norms_sq)
      - spins_array: numpy array shape (num_samples, m) with spins ±1 (or 0/1)
      - labels: labels order returned by dicts_to_spin_list (not used for decoding here,
                only required that spins_array columns match the bit-index order expected by meta)
      - meta: your meta dict returned by build_meta_with_fixed(...) (used by decode_bits_to_u_with_fixed)
      - B: basis matrix used to compute v (same B used when building meta)
    Returns:
      U: (num_samples, n) integer coefficient vectors u
      V: (num_samples, n) lattice vectors v
      norms_sq: (num_samples,) squared Euclidean norms of v
    """
    # 1) bits x in {0,1} from spins
    X = np.where(spins_array == -1, 0, spins_array)   # shape (num_samples, m)
    num_samples = X.shape[0]
    n = meta['n']
    U_list = []
    V_list = []
    for i in range(num_samples):
        x_bits = X[i, :]
        # decode into u using your existing function (handles fixed coords)
        u = decode_bits_to_u_with_fixed(x_bits, meta)   # returns length-n vector
        # form v according to convention
        if meta.get('convention', 'rows') == 'rows':
            v = (u @ B).astype(np.int64)
        else:
            v = (B @ u).astype(np.int64)
        U_list.append(u.astype(np.int64))
        V_list.append(v.astype(np.int64))
    U = np.vstack(U_list)   # shape (num_samples, n)
    V = np.vstack(V_list)   # shape (num_samples, n)
    norms_sq = np.sum(V.astype(np.int64)**2, axis=1)
    return U, V, norms_sq

def infer_order_from_keys(sample_dict: Dict[Any, Any]) -> List[Any]:
    """
    Choose an ordering of keys for conversion.
    If keys look like integer labels (strings or ints), returns them sorted by int(key).
    Otherwise returns sorted by string(key).
    """
    keys = list(sample_dict.keys())
    # try numeric sort (works when keys are like '1','2', ... or 1,2,...)
    try:
        keyed = [(int(k), k) for k in keys]
        keyed.sort(key=lambda t: t[0])
        ordered_keys = [orig for (_, orig) in keyed]
        return ordered_keys
    except Exception:
        # fallback to string sort (stable)
        return sorted(keys, key=lambda k: str(k))

def dicts_to_spin_list(sample_dicts: Iterable[Dict[Any, Any]],
                       key_order: List[Any] = None) -> Tuple[np.ndarray, List[Any]]:
    """
    Convert iterable of sample dicts -> (spins_array, labels_order)
    - sample_dicts: iterable of dict-like maps variable_label -> spin (±1, np.int8, etc.)
    - key_order: optional list specifying desired order of labels. If None, infer from first dict.
    Returns:
      spins: numpy array shape (num_samples, N) of ints (+1/-1)
      labels_order: list of labels (same type as keys in dicts) in the columns order
    """
    sample_list = list(sample_dicts)
    if len(sample_list) == 0:
        return np.zeros((0,0), dtype=int), []

    if key_order is None:
        labels = infer_order_from_keys(sample_list[0])
    else:
        labels = list(key_order)

    rows = []
    for d in sample_list:
        row = []
        for k in labels:
            # accept k as original type or as stringified int (try both)
            if k in d:
                val = d[k]
            else:
                # try converting label types (useful when keys are strings but passed ints etc.)
                try:
                    val = d[str(k)]
                except Exception:
                    try:
                        val = d[int(k)]
                    except Exception:
                        # missing key -> fallback 0 (or raise?)
                        val = 0
            # coerce numpy scalars to python int (+1/-1)
            row.append(int(val))
        rows.append(row)

    spins = np.array(rows, dtype=int)
    return spins, labels
def ising_energy(s, J, h=None):
    """Compute Ising energy H(s) = -0.5 s^T J s - h^T s."""
    if h is None:
        h = np.zeros(len(s))
    return -0.5 * s @ J @ s - h @ s


def generate_random_lattice_vectors(basis, num_vectors, max_coeff=50):
    """Generates a list of random vectors belonging to the lattice."""
    dimension = basis.shape[0]
    vectors = []
    for _ in range(num_vectors):
        coeffs = np.random.randint(-max_coeff, max_coeff + 1, size=dimension)
        vector = np.dot(basis.T, coeffs)
        vectors.append(vector)
    return vectors

def is_valid_lattice_vector(vector, basis, tol=1e-9):
    """Checks if a vector is a valid member of the lattice."""
    try:
        coeffs = np.linalg.solve(basis.T, vector)
        return np.all(np.abs(coeffs - np.round(coeffs)) < tol)
    except np.linalg.LinAlgError:
        return False

def gauss_sieve(basis, initial_vectors):
    """
    Performs a Gauss Sieve on a list of lattice vectors.

    Args:
        basis (np.ndarray): The lattice basis.
        initial_vectors (list[np.ndarray]): The starting list of vectors.

    Returns:
        tuple: A tuple containing:
            - list[np.ndarray]: The final list of processed, short vectors.
            - dict: A dictionary containing the history of norms for plotting.
    """
    history = {'min_norm': [], 'max_norm': [], 'avg_norm': []}
    
    # --- 1. Initialization ---
    processed_list = []
    # Sort initial vectors by squared norm and put them in an efficient queue
    initial_vectors.sort(key=lambda v: np.dot(v,v))
    # print(initial_vectors)
    queue = deque(initial_vectors)

    print("\n--- Starting Gauss Sieve ---")
    processed_count = 0

    # --- 2. Main Loop ---
    while queue:
        # a. Get the shortest candidate from the queue
        v_candidate = queue.popleft()
        candidate_norm = np.dot(v_candidate, v_candidate)
        
        is_reduced = False

        # b. The Trial: Test against all processed "tools"
        for v_tool in processed_list:
            # c. The Test
            diff = v_candidate - v_tool
            diff_norm = np.dot(diff, diff)

            vec_sum = v_candidate + v_tool
            sum_norm = np.dot(vec_sum, vec_sum) # Corrected Line
            
            # d. The Crucial Question
            if diff_norm > 0 and diff_norm < candidate_norm:
                # Outcome A: Reduction Found. Discard candidate, add new vector to queue.
                queue.append(diff)
                is_reduced = True
                break # Stop testing this candidate
            
            if sum_norm > 0 and sum_norm < candidate_norm:
                # Outcome A: Reduction Found.
                queue.append(vec_sum)
                is_reduced = True
                break # Stop testing this candidate

        # --- 3. The Fate of the Candidate ---
        if not is_reduced:
            # Outcome B: No reduction found. Promote the candidate.
            processed_list.append(v_candidate)
            processed_count += 1
            print(f"\rVectors Processed: {processed_count}", end="")

            # --- For Plotting: Record the state of the system ---
            all_current_vectors = processed_list + list(queue)
            if all_current_vectors:
                norms = [np.dot(v, v) for v in all_current_vectors]
                history['min_norm'].append(np.min(norms))
                history['max_norm'].append(np.max(norms))
                history['avg_norm'].append(np.mean(norms))
        else:
            # A reduction occurred. Re-sort the queue to keep it optimal.
            sorted_queue = sorted(list(queue), key=lambda v: np.dot(v,v))
            queue = deque(sorted_queue)

    print("\n--- Sieve Finished ---")
    return processed_list, history

def gauss_sieve2(basis, initial_vectors):
    """
    Performs a Gauss Sieve on a list (or ndarray) of lattice vectors.

    Accepts initial_vectors as:
      - a Python list of 1-D array-like vectors, or
      - a NumPy 2-D array of shape (num_vectors, dim).

    Returns:
      processed_list, history
    """
    history = {'min_norm': [], 'max_norm': [], 'avg_norm': []}

    # --- normalize initial_vectors to a Python list of 1-D np.int64 arrays ---
    if isinstance(initial_vectors, np.ndarray):
        if initial_vectors.ndim != 2:
            raise ValueError("If initial_vectors is ndarray it must be 2-D (num_vectors, dim).")
        initial_vectors_list = [np.asarray(initial_vectors[i], dtype=np.int64).ravel() for i in range(initial_vectors.shape[0])]
    else:
        # Coerce any iterable-of-iterables into list of 1-D arrays
        initial_vectors_list = [np.asarray(v, dtype=np.int64).ravel() for v in list(initial_vectors)]

    # Remove any zero-length vectors accidentally present
    initial_vectors_list = [v for v in initial_vectors_list if v.size > 0]

    # --- sort by squared norm (shortest first) using Python list sort with key ---
    initial_vectors_list.sort(key=lambda v: int(np.dot(v, v)))

    queue = deque(initial_vectors_list)
    processed_list = []
    print("\n--- Starting Gauss Sieve ---")
    processed_count = 0

    # --- main loop ---
    while queue:
        v_candidate = queue.popleft()
        candidate_norm = int(np.dot(v_candidate, v_candidate))

        is_reduced = False

        for v_tool in processed_list:
            diff = v_candidate - v_tool
            diff_norm = int(np.dot(diff, diff))

            vec_sum = v_candidate + v_tool
            sum_norm = int(np.dot(vec_sum, vec_sum))

            # reduction found
            if 0 < diff_norm < candidate_norm:
                queue.append(diff.astype(np.int64))
                is_reduced = True
                break
            if 0 < sum_norm < candidate_norm:
                queue.append(vec_sum.astype(np.int64))
                is_reduced = True
                break

        if not is_reduced:
            processed_list.append(v_candidate.astype(np.int64))
            processed_count += 1
            # progress print (overwrites same line)
            print(f"\rVectors Processed: {processed_count}", end="")

            # record stats for plotting
            all_current = processed_list + list(queue)
            if all_current:
                norms = [int(np.dot(v, v)) for v in all_current]
                history['min_norm'].append(np.min(norms))
                history['max_norm'].append(np.max(norms))
                history['avg_norm'].append(float(np.mean(norms)))
        else:
            # re-sort queue to keep the next popped candidate the shortest
            sorted_queue = sorted(list(queue), key=lambda v: int(np.dot(v, v)))
            queue = deque(sorted_queue)

    print("\n--- Sieve Finished ---")
    return processed_list, history

def plot_sieving_progress(history):
    """Plots the sieving progress. The x-axis is now 'Vectors Processed'."""
    print("\nGenerating plot of the sieving process...")
    processed_steps = range(len(history['min_norm']))
    plt.figure(figsize=(12, 7))
    plt.plot(processed_steps, history['max_norm'], label='Max Squared Norm', color='red', alpha=0.8)
    plt.plot(processed_steps, history['avg_norm'], label='Average Squared Norm', color='orange', linestyle='--')
    plt.plot(processed_steps, history['min_norm'], label='Min Squared Norm', color='green', linewidth=2)
    plt.title('Sieve Process', fontsize=16)
    plt.xlabel('steps', fontsize=12)
    plt.ylabel('Squared Norm (Log Scale)', fontsize=12)
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.show()

def lll_reduce_basis(B):
    """
    Performs LLL reduction on a matrix B and returns the reduced matrix.
    
    Parameters:
        B (np.ndarray): Input matrix.
        
    Returns:
        np.ndarray: LLL-reduced matrix.
    """
    B_int = B.astype(int).tolist()
    A = IntegerMatrix.from_matrix(B_int)
    LLL.reduction(A)
    return np.array(list(A), dtype=int)

def build_meta_with_fixed(B: np.ndarray, bits: int = 3, fixed_idx: int = None, fixed_val: int = 1, convention: str = "rows") -> Dict[str, Any]:
    """
    Build meta for signed-binary encoding, allowing one (or more) coordinates to be fixed.
    convention: 'rows' => v = u @ B, G = B @ B.T
                'columns' => v = B @ u, G = B.T @ B
    fixed_idx: index (0-based) of coordinate to fix, or list/tuple of indices. If None, no fixed coords.
    fixed_val: integer value to assign to fixed coordinate(s); if fixed_idx is sequence, can be int or sequence.
    Returns meta dict with var_index_map only for free coords, and 'fixed' mapping.
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

    # normalize fixed_idx to list
    if fixed_idx is None:
        fixed_indices = []
        fixed_vals = {}
    else:
        if isinstance(fixed_idx, int):
            fixed_indices = [fixed_idx]
            fixed_vals = {fixed_idx: int(fixed_val)}
        else:
            fixed_indices = list(fixed_idx)
            if hasattr(fixed_val, '__iter__') and not isinstance(fixed_val, (str,bytes)):
                fixed_vals = {i: int(v) for i, v in zip(fixed_indices, fixed_val)}
            else:
                fixed_vals = {i: int(fixed_val) for i in fixed_indices}

    # build variable index map only for free coordinates
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
        bit_weights[(i, bits - 1)] = -(1 << (bits - 1))
        var_idx += 1

    m = var_idx
    meta = {
        'n': n,
        'bits': bits,
        'var_index_map': var_index_map,
        'bit_weights': bit_weights,
        'm': m,
        'G': G,
        'convention': convention,
        'fixed_indices': fixed_indices,
        'fixed_vals': fixed_vals
    }
    return meta

def decode_bits_to_u_with_fixed(x_bits: np.ndarray, meta: Dict[str, Any]) -> np.ndarray:
    """
    Decode bits to full u vector including fixed entries from meta.
    x_bits length = meta['m'].
    """
    n = meta['n']
    bits = meta['bits']
    var_map = meta['var_index_map']
    bit_weights = meta['bit_weights']
    u = np.zeros(n, dtype=np.int64)
    # fill free coords
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

def enumerate_all_with_fixed(B: np.ndarray, meta: Dict[str, Any]) -> List[Tuple[int, List[int], List[int], List[int]]]:
    """
    Enumerate all 2^m configs for free bits, produce (energy, x_bits_list, u_list, v_list).
    """
    m = meta['m']
    G = meta['G'].astype(np.int64)
    conv = meta['convention']
    results = []
    for bits_tuple in itertools.product([0,1], repeat=m):
        x = np.array(bits_tuple, dtype=np.int64)
        u = decode_bits_to_u_with_fixed(x, meta)
        if conv == "columns":
            v = (B @ u).astype(np.int64)
        else:
            v = (u @ B).astype(np.int64)
        energy = int(u @ (G @ u))
        results.append((energy, list(x.tolist()), list(u.tolist()), list(v.tolist())))
    return results

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
# Example: use user's B3 and fix last coordinate (index 2) to 1
def qubo_ising(Q):
    total_qubits = Q[0].shape
    total_qubits = total_qubits[0]
    # Step 1: Convert QUBO matrix to dictionary format
    Q_dict = {(f'{i+1}', f'{j+1}'): Q[i, j] for i in range(total_qubits) for j in range(total_qubits) if Q[i, j] != 0}

    # Step 2: Convert QUBO to Ising
    h_dict, J_dict, offset = dimod.qubo_to_ising(Q_dict)

    # Step 3: Extract h and J arrays
    h = np.zeros(total_qubits)
    J = np.zeros((total_qubits, total_qubits))
    for i in range(int(total_qubits)):
       for j in range(int(total_qubits)):
          if i != j:
             J[i,j] = J_dict.get(('{}'.format(i+1), '{}'.format(j+1)))
          else:
             h[i] = h_dict.get(('{}'.format(i+1)))

    return Q_dict, h_dict, J_dict, Q, h, J, offset



def brute_force_ising(J, h=None, max_enumeration_bits=24):
    """
    Brute-force minimization of the classical Ising Hamiltonian.
    Inputs:
      J : (N,N) symmetric matrix of couplings (diagonal will be ignored)
      h : (N,) external fields (defaults to zero)
      max_enumeration_bits: safety cutoff; will raise error if 2^N too large
    Returns:
      dict with keys:
        'ground_energy' : minimum energy found (float)
        'ground_states' : array of shape (k,N) with spin vectors (+1/-1) tied for min
        'all_energies'   : (optional) energies array for all states (not returned to save memory)
        'N' : number of spins
        'num_states_searched' : 2**N
    """
    J = np.asarray(J, dtype=float)
    if J.ndim != 2 or J.shape[0] != J.shape[1]:
        raise ValueError("J must be a square (N,N) matrix.")
    N = J.shape[0]
    if h is None:
        h = np.zeros(N, dtype=float)
    else:
        h = np.asarray(h, dtype=float)
        if h.shape != (N,):
            raise ValueError("h must be of shape (N,)")

    if N > max_enumeration_bits:
        raise MemoryError(f"N={N} too large for brute-force (2^{N} states). Increase max_enumeration_bits with caution.")

    M = 1 << N  # number of configurations 2**N
    # Build spins matrix S of shape (M, N) where each row is a spin config in {+1,-1}
    # We use bit operations to avoid Python-level loops for each configuration.
    # For N moderately small this is efficient. Beware memory: M*N elements.
    # We'll construct in int8 to save memory then cast to float for energy calc.
    indices = np.arange(M, dtype=np.uint32)
    # Create bit mask positions for each spin index 0..N-1
    bit_positions = (1 << np.arange(N, dtype=np.uint32))
    # boolean array shape (M, N): True where bit is 1
    bits = ((indices[:, None] & bit_positions[None, :]) != 0)
    # Map bits {0,1} -> spins {+1,-1} via s = 1 - 2*bit
    S = (1 - 2 * bits).astype(np.int8)  # shape (M, N)

    # Compute energies: E = -0.5 * s^T J s - h . s
    # Use vectorized einsum for speed
    # Note: J should have diagonal zero or otherwise diagonal contributes J_ii (we divide by 2)
    SJ = S.dot(J)               # shape (M, N)
    quad = np.einsum('mi,mi->m', SJ, S)  # s @ J @ s for each row (includes double counting)
    energies = -0.5 * quad - S.dot(h)   # shape (M,)

    minE = energies.min()
    min_idx = np.nonzero(np.isclose(energies, minE))[0]
    ground_states = S[min_idx].astype(int)  # convert to python int type for readability

    return {
        'ground_energy': float(minE),
        'ground_states': ground_states,
        'N': N,
        'num_states_searched': int(M),
    }



def s0_to_initial_states(s0, hdict):
    """
    Convert an s0 spin vector (±1) into the `initial_states` format expected by
    dimod/dwave samplers: a list of dicts mapping variable-label -> spin.

    Args:
        s0: 1D array-like of shape (N,) or 2D array-like (num_states, N) with spins in {-1, +1}.
        hdict: the h dictionary/array you pass to sample_ising (used to infer variable labels).
               Can be a dict with keys like '1','2' or 1,2 or a numpy/list (then labels 0..N-1 are used).

    Returns:
        initial_states: list of dicts, each dict maps the sampler variable-labels to spins.
    """
    arr = np.asarray(s0)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)  # make (1, N)
    num_states, N = arr.shape

    # Infer variable labels from hdict while preserving original label types
    if isinstance(hdict, (list, tuple, np.ndarray)):
        labels = list(range(len(hdict)))
    elif isinstance(hdict, dict):
        labels = list(hdict.keys())
        # Try to sort labels numerically when possible (preserve original keys in mapping)
        try:
            # build (int_value, original_label) tuples
            mapped = [(int(k), k) for k in labels]
            mapped.sort(key=lambda x: x[0])
            labels = [orig for (_, orig) in mapped]
        except Exception:
            # fallback: string-sorted order of labels
            labels = sorted(labels, key=lambda x: str(x))
    else:
        raise TypeError("hdict must be dict or array-like (list/np.array)")

    if len(labels) != N:
        raise ValueError(f"Length mismatch: inferred {len(labels)} labels from hdict but s0 has length {N}.")

    # Build list of dicts for initial_states
    initial_states = []
    for row in arr:
        state_dict = {labels[i]: int(row[i]) for i in range(N)}
        initial_states.append(state_dict)

    return initial_states

def run_cim(B, meta, NUM_VECTORS=2000, iters=1000, dt=0.4):
    """Run CIM pipeline on lattice basis B with given meta encoding."""
    Q, lin, const = build_qubo_full(meta)
    Q_single, c_single = qubo_single_from_full(Q, lin, const)
    Qdict, hdict, Jdict, Q, h, J, offset = qubo_ising(Q_single)
    offset += c_single

    # clean NaNs
    J = np.nan_to_num(J)
    h = np.nan_to_num(h)
    J = 2 * J

    # run CIM
    solver = CFC(-J, h=-h, batch_size=NUM_VECTORS, n_iter=iters, dt=dt, backend='gpu-float32')
    solver.update()
    energies, spins = solver.calc_energy()
    sorted_energy, indices = torch.sort(energies)
    sorted_spins = spins[:, indices].T
    numpy_sorted_spins = sorted_spins.cpu().numpy()

    # take best spin config
    best_spin = numpy_sorted_spins[0, 0].astype(int)
    s0 = np.array(best_spin)

    # greedy descent refinement
    solver_greedy = SteepestDescentSampler()
    initial_states = s0_to_initial_states(s0, hdict)
    sampleset = solver_greedy.sample_ising(hdict, Jdict, initial_states=initial_states, num_reads=NUM_VECTORS)

    # convert back to lattice vectors
    sumple = [i for i in sampleset.samples()]
    spins_array, labels = dicts_to_spin_list(sumple)
    U, V, norms_sq = spins_array_to_u_v(spins_array, labels, meta, B)
    return V, norms_sq.min()

def experiment(n_range=range(2, 21), NUM_VECTORS=2000, bits=4):
    results_random = []
    results_cim = []
    results_lll_cim = []

    for n in n_range:
        print(f"\n=== LWE Dimension n={n} ===")

        # load A, b, s, e for this n
        A = np.load(f"Instances/A{n}.npy")
        b = np.load(f"Instances/b{n}.npy")
        Bai = primal(A, b)

        # --- Random lattice vectors ---
        rand_vecs = generate_random_lattice_vectors(Bai, NUM_VECTORS)
        norms = [np.dot(v, v) for v in rand_vecs if is_valid_lattice_vector(v, Bai)]
        shortest_rand = np.min(norms)
        results_random.append(shortest_rand)
        print(f"Random shortest norm: {shortest_rand}")

        # --- CIM on original basis ---
        fixed_idx = Bai.shape[0] - 1
        meta = build_meta_with_fixed3(Bai, bits=bits, fixed_idx=fixed_idx, fixed_val=1, convention="rows")
        V_cim, shortest_cim = run_cim(Bai, meta, NUM_VECTORS=NUM_VECTORS)
        results_cim.append(shortest_cim)
        print(f"CIM shortest norm: {shortest_cim}")

        # --- CIM on LLL-reduced basis ---
        Bai_lll = lll_reduce_basis(Bai)
        fixed_idx = Bai_lll.shape[0] - 1
        meta_lll = build_meta_with_fixed3(Bai_lll, bits=bits, fixed_idx=fixed_idx, fixed_val=1, convention="rows")
        V_cim_lll, shortest_cim_lll = run_cim(Bai_lll, meta_lll, NUM_VECTORS=NUM_VECTORS)
        results_lll_cim.append(shortest_cim_lll)
        print(f"LLL+CIM shortest norm: {shortest_cim_lll}")

    return results_random, results_cim, results_lll_cim

def simulate_snn_cim(J, 
                     alpha=1.0,    
                     beta=0.3,     
                     epsilon=0.1,  
                     g=1.0,        
                     n_iter=200, 
                     dt=0.01,
                     x0=None, k0=None):
    """
    Simulate SNN-CIM ODEs (Eq.1 of the paper) via Euler method.
    Returns x(t), k(t), and energy(t)
    """
    N = J.shape[0]
    x = np.zeros(N) if x0 is None else x0.copy()
    k = np.zeros(N) if k0 is None else k0.copy()
    x += 0.01 * np.random.randn(N)
    k += 0.01 * np.random.randn(N)

    J_xk =  g
    J_kx = -g

    x_hist = np.zeros((n_iter, N))
    k_hist = np.zeros((n_iter, N))
    energy_hist = np.zeros(n_iter)

    for t in range(n_iter):
        I_ext = np.tanh(epsilon * (J @ x))

        dx = alpha * x - x**3 + J_xk * beta * k + I_ext
        dk = -beta * k + J_kx * x

        x += dt * dx
        k += dt * dk

        x_hist[t] = x
        k_hist[t] = k

        # spin: sign(x)
        spin = np.sign(x)
        energy_hist[t] = -0.5 * spin @ J @ spin

    return x_hist, k_hist, energy_hist

if __name__ == "__main__":


        # Run experiment
    rand_norms, cim_norms, lll_cim_norms = experiment()

    # Plot
    dims = list(range(2, 21))
    plt.figure(figsize=(10,6))
    plt.plot(dims, rand_norms, 'o-', label="Random sampling")
    plt.plot(dims, cim_norms, 's-', label="CIM sampling")
    plt.plot(dims, lll_cim_norms, 'd-', label="LLL + CIM sampling")
    plt.yscale("log")
    plt.xlabel("LWE Dimension n")
    plt.ylabel("Shortest norm found")
    plt.title("Comparison of Sampling Methods for LWE Lattices (n=2..20)")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.show()
    NUM_VECTORS = 10000

    s = np.load("Instances/s17.npy")
    # print(s)

    A = np.load("Instances/A17.npy")
    # print(A)

    b = np.load("Instances/b17.npy")
    # print(b)

    e = np.load("Instances/e17.npy")
    # print(e)
    # print((s@A+e-b)/3329)
    Bai = primal(A, b)
    # det1 = np.sqrt(abs(np.linalg.det(Bai@Bai.T)))
    # print("det of Bai lattice without LLL",det1)
    # Bai = lll_reduce_basis(Bai)
    # det1 = np.sqrt(abs(np.linalg.det(Bai@Bai.T)))
    # print("det of Bai lattice without LLL",det1)
    # print(Bai[39]@Bai[39])
    B3 = Bai

    bits = 4
    n = B3.shape[0]
    fixed_idx = n - 1
    fixed_val = 1

    meta = build_meta_with_fixed3(B3, bits=bits, fixed_idx=fixed_idx, fixed_val=fixed_val, convention="rows")

    Q, lin, const = build_qubo_full(meta)



    Q_single, c_single = qubo_single_from_full(Q, lin, const)
    # print(Q_single)

    Qdict, hdict, Jdict, Q, h, J, ofset = qubo_ising(Q_single)

    ofset = ofset + c_single

    # print(ofset)
    # print(h)

    where_are_NaNs = np.isnan(J)
    J[where_are_NaNs] = 0

    where_are_NaNs = np.isnan(h)
    h[where_are_NaNs] = 0

    J = 2 * J
    solver_greedy = SteepestDescentSampler()
    for _ in range(1):


        solver = CFC(-J, h=-h, batch_size=1000, n_iter=1000, dt=0.4, backend='gpu-float32')
        # print(out['ground_energy']+ofset)
        # print(out['ground_states'])
        solver.update()

        # Compute and display results
        energies, spins = solver.calc_energy()
        sorted_energy, indices = torch.sort(energies)
        sorted_spins = spins[:, indices]
        sorted_spins = sorted_spins.T
        numpy_sorted_spins = sorted_spins.cpu().numpy()

        s1 = numpy_sorted_spins[:,0]

        # print('Energy values:', sorted_energy[0,0]+ofset)
        # print('Spin configurations (signs):', sorted_spins[0,0])
        sX = sorted_spins[0,0]
        bestS = []
        # print(sX.shape)

        for i in sX:
            bestS.append(int(i))

        bestSnp = np.array(bestS)

        # print(bestSnp)

        s0 = bestSnp
        initial_states = s0_to_initial_states(s0, hdict)
        # print(initial_states)
        sampleset = solver_greedy.sample_ising(hdict, Jdict,
                                               initial_states=initial_states,
                                          num_reads=NUM_VECTORS)
        best_sample = sampleset.first.sample
        best_energy = float(sampleset.first.energy)
        best_samples = sampleset.first.sample
        sumple = []
        print("check1", best_energy+ofset)
        # print("check", best_sample)
        # print(sampleset)
        for i in sampleset.samples():
            sumple.append(i)
            # print(i)
        # print(sumple[0])
        spins_array, labels = dicts_to_spin_list(sumple)
        # print("labels order:", labels)
        # print("spins_array shape:", spins_array.shape)
        # print(spins_array)
        # spins_array shape (num_samples, m) from dicts_to_spin_list(...)
        U, V, norms_sq = spins_array_to_u_v(spins_array, labels, meta, B3)
        print(type(V))
        # Print top few results sorted by norm
        order = np.argsort(norms_sq)
        for idx in order[:5]:
            print(f"v = {V[idx].tolist()}")
        initial_norms = [np.dot(v, v) for v in V]
        print(f"\nInitial shortest vector (squared norm): {min(initial_norms):.2f}")

        final_processed_list, sieve_history = gauss_sieve2(Bai, V)

        final_norms = [np.dot(v, v) for v in final_processed_list]
        shortest_vector = final_processed_list[np.argmin(final_norms)]
        print(type(shortest_vector))
        print(f"\nFinal shortest vector (squared norm): {min(final_norms):.2f}")
        print("\nShortest vector found by the sieve:\n", shortest_vector)
        plot_sieving_progress(sieve_history)

        minGCIM = sieve_history['min_norm']
        minGCIM = np.array(minGCIM)
        np.save("shorts_of_SieveCIM13",minGCIM)

        # B_new = np.vstack((shortest_vector, Bai))  

        # B_new = lll_reduce_basis(B_new)
        # B_new = B_new[1:]     
        # print(B_new) 
        # det2 = np.sqrt(np.linalg.det(B_new@B_new.T))
        # print("det of Bai lattice with LLL and inserting vector",det2)

        # NUM_VECTORS = NUM_VECTORS + 1000
        # B3 = B_new

        # bits = 4
        # n = B3.shape[0]
        # fixed_idx = n - 1
        # fixed_val = 1

        # meta = build_meta_with_fixed3(B3, bits=bits, fixed_idx=fixed_idx, fixed_val=fixed_val, convention="rows")

        # Q, lin, const = build_qubo_full(meta)



        # Q_single, c_single = qubo_single_from_full(Q, lin, const)
        # # print(Q_single)

        # Qdict, hdict, Jdict, Q, h, J, ofset = qubo_ising(Q_single)

        # ofset = ofset + c_single

        # # print(ofset)
        # # print(h)

        # where_are_NaNs = np.isnan(J)
        # J[where_are_NaNs] = 0

        # where_are_NaNs = np.isnan(h)
        # h[where_are_NaNs] = 0

        # J = 2 * J



        # print("check", best_sample)
        # print(sampleset)
    initial_vector_list = generate_random_lattice_vectors(Bai, NUM_VECTORS)

    # --- 2. VALIDITY CHECK ---
    if not all(is_valid_lattice_vector(v, Bai) for v in initial_vector_list):
        print("Error! An initial vector was found to be invalid.")
        exit()
    print("\nSuccess! All initial vectors are valid members of the lattice.")

    # --- 3. RUN THE GAUSS SIEVE ---
    initial_norms = [np.dot(v, v) for v in initial_vector_list]
    print(f"\nInitial shortest vector (squared norm): {min(initial_norms):.2f}")

    final_processed_list, sieve_history = gauss_sieve(Bai, initial_vector_list)

    # --- 4. SHOW RESULTS ---
    final_norms = [np.dot(v, v) for v in final_processed_list]
    shortest_vector = final_processed_list[np.argmin(final_norms)]
    
    print(f"\nFinal shortest vector (squared norm): {min(final_norms):.2f}")
    print("\nShortest vector found by the sieve:\n", shortest_vector)


    m, n = Bai.shape
    M = IntegerMatrix(m, n)
    for i in range(m):
        for j in range(n):
            M[i, j] = int(Bai[i, j])


    shortest_vector = SVP.shortest_vector(M, method="fast", pruning=None)

    print(shortest_vector)


    # --- 5. PLOT THE RESULTS ---
    plot_sieving_progress(sieve_history)
    minGCIM = sieve_history['min_norm']
    minGCIM = np.array(minGCIM)
    np.save("shorts_of_SieveCRandom13",minGCIM)


