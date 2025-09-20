# Gauss-Sieve CIM

**Summary**

This repository contains helper functions for: building/decoding signed-binary encodings of lattice vectors, running simple Gauss-sieve routines, converting lattice vectors to Ising/QUBO representations, running a Coherent Ising Machine (CIM) pipeline, and several utilities (LLL wrap, plotting, brute-force Ising). The code is intended to support experimentation on lattice problems (e.g. LWE/primal lattices) using Ising solvers and CIM-style heuristics.

---

## Contents of this README

1. Quick overview
2. Requirements & installation
3. Important conventions (rows vs columns, encoding)
4. Key functions (API reference + examples)
5. Typical workflows & examples
6. Performance & safety notes
7. License & author

---

## 1) Quick overview
- The main file is IsingLWE.py
- `build_meta_with_fixed(B, bits, fixed_idx, fixed_val, convention)` — build a `meta` descriptor for signed-binary encoding of integer coefficient vectors `u` (with the option to fix coordinate(s)).
- `decode_bits_to_u_with_fixed(x_bits, meta)` — decode an m-bit vector into full integer `u` including fixed entries.
- `build_all_coord_lookups(meta)` / `_build_value_to_bits_lookup_for_coord` — internal helpers to map integer `u[i]` values to bit patterns for each free coordinate.
- `u_to_bits_vector` / `v_to_u_from_B` — convert between `u`, bits `x`, and lattice vector `v` (via basis `B`).
- `gauss_sieve` / `gauss_sieve2` — two versions of a basic Gauss-sieve implementation returning a list of short lattice vectors and a `history` dict for plotting.
- `plot_sieving_progress` — visualize min/avg/max squared norms during the sieve.
- `lll_reduce_basis` — wrapper around `fpylll.LLL` to return an LLL-reduced integer matrix.
- `gauss_vs_to_ising_states` — **high level**: converts Gauss-sieve output vectors `V` into bit vectors, spins and Ising dicts, along with `u` and norms.
- `IsingLWE` — pipeline: build QUBO from `meta`, convert to Ising, run CIM (via `IsingMachine.CFC`) and refine with `SteepestDescentSampler` from dwave-package. Returns candidate lattice vectors and best norm.
- `simulate_snn_cim` — a simple ODE-based SNN-CIM Euler integrator useful for prototyping.
- `brute_force_ising` — exact minimizer of small Ising Hamiltonians (warning: exponential in N).
- `enumerate_all_with_fixed` — exact enumeration of all `2^m` bit patterns with fixed coordinates; useful for small instances and debug.
- Utilities for converting between dict-samples and arrays: `dicts_to_spin_list`, `bits_to_spin_dicts`, `spins_array_to_u_v`, `s0_to_initial_states`, `infer_order_from_keys`.

---

## 2) Requirements & installation

Minimum required libraries (approx):
- Python 3.8+
- numpy
- matplotlib
- fpylll
- dimod
- dwave-ocean-sdk (for `dwave.samplers.SteepestDescentSampler`) — optional if you skip the refinement step
- `singlequbo` and `IsingMachine` — local modules used for QUBO assembly and CIM solver (`CFC`).
- PyTorch (used indirectly by the CFC solver in `run_cim`) if the solver implementation uses it.

Install with pip (example):

```bash
pip install numpy matplotlib fpylll dimod dwave-ocean-sdk
# plus local requirements for IsingMachine and singlequbo
```

> Note: `fpylll` may require system-level dependencies. Install via conda if preferred: `conda install -c conda-forge fpylll`.

---

## 3) Important conventions and shapes

- `B` (basis): if `convention == 'rows'`, code assumes row-basis where lattice vector `v = u @ B` and Gram `G = B @ B.T`. If `convention == 'columns'`, `v = B @ u` and `G = B.T @ B`.
- `meta` has keys: `n` (dimension), `bits` (bits per free coord), `var_index_map` mapping `(i,k)` to bit-index, `bit_weights` giving signed weights per bit, `m` (total free bits), `G`, `convention`, `fixed_indices`, `fixed_vals`.
- `x_bits` is an array of length `m` with values in `{0,1}`. Conversion to spins: `s = 1 - 2*x_bits` (so `x=0 -> s=+1`, `x=1 -> s=-1`).
- `U` is integer vector length `n`. `V` is lattice vector length `n`.

---

## 4) Key functions — API + examples

### `build_meta_with_fixed(B, bits=3, fixed_idx=None, fixed_val=1, convention='rows')`
Returns: `meta` dict.

Example:
```python
meta = build_meta_with_fixed(B, bits=4, fixed_idx=B.shape[0]-1, fixed_val=1, convention='rows')
```
This fixes the last coordinate to 1 and encodes the others using `bits` signed-binary (MSB negative) per coordinate.

---

### `decode_bits_to_u_with_fixed(x_bits, meta)`
Input: `x_bits` length `m` -> returns full integer `u` length `n`.

Example:
```python
u = decode_bits_to_u_with_fixed(x_bits, meta)
v = (u @ B)  # if convention rows
```

---

### `v_to_u_from_B(v, B, convention='rows', tol=1e-6)`
Recover integer `u` from `v` and `B`. Returns `(u_rounded, ok_flag)`.

Example:
```python
u, ok = v_to_u_from_B(v, B, 'rows')
```

---

### `u_to_bits_vector(u, meta, lookups)`
Convert full `u` to `x_bits` using precomputed coordinate lookups (from `build_all_coord_lookups`). Returns `(x_bits, ok)` where `ok==False` if some coordinate value is not representable by available bits.

Example:
```python
lookups = build_all_coord_lookups(meta)
x_bits, ok = u_to_bits_vector(u, meta, lookups)
```

---

### `gauss_sieve2(basis, initial_vectors)`
A robust version that accepts ndarray or list and returns `(processed_list, history)` where `processed_list` is list of short vectors and `history` contains `min_norm`, `max_norm`, `avg_norm` per step.

Example:
```python
processed, history = gauss_sieve2(B, initial_vectors)
plot_sieving_progress(history)
```

---

### `gauss_vs_to_ising_states(V, B, meta, label_type='str')`
High-level conversion used after running a Gauss sieve. Accepts `V` shape `(num, n)` or list. Returns dictionary with `X_bits`, `S_spins`, `ising_dicts`, `U`, `V`, `norms_sq`, `valid_mask`, `lookups`.

Example:
```python
out = gauss_vs_to_ising_states(processed, B, meta)
print(out['X_bits'].shape, out['ising_dicts'][0])
```

---

### `run_cim(B, meta, NUM_VECTORS=2000, iters=1000, dt=0.4)`
Builds QUBO from `meta` and runs the CIM solver pipeline (`IsingMachine.CFC`) and a steepest-descent refinement. Returns `(V_candidates, best_norm)`.

Minimal usage:
```python
V, best_norm = run_cim(B, meta, NUM_VECTORS=500, iters=500)
```

> **Important:** `run_cim` depends on `singlequbo.build_qubo_full`, `IsingMachine.CFC` and a working GPU setup if the solver uses GPU. If you don't have these, either stub them or replace the solver call with another sampler.

---

### `brute_force_ising(J, h=None, max_enumeration_bits=24)`
Exactly finds ground states for small `N`. **Do not** use for `N > ~24` unless you have lots of memory/cpu.

Example:
```python
res = brute_force_ising(J, h)
print(res['ground_energy'], res['ground_states'].shape)
```

---

### `simulate_snn_cim(J, alpha, beta, epsilon, g, n_iter, dt, x0, k0)`
Euler integration of a coupled ODE system modeling a SNN-CIM. Returns `(x_hist, k_hist, energy_hist)` useful for diagnostics/plots.

Example:
```python
x_hist, k_hist, E = simulate_snn_cim(J, n_iter=200, dt=0.01)
plt.plot(E)
```

---

## 5) Typical workflows

**A) Small exact enumeration (debug)**
1. Build `meta` with `bits` small (e.g. 3). 2. Use `enumerate_all_with_fixed(B, meta)` to list all (`2^m`) configurations and energies.

**B) Gauss-sieve → convert → Ising → CIM**
1. Prepare `initial_vectors` for the lattice (random or structured). 2. `processed, history = gauss_sieve2(B, initial_vectors)`. 3. `meta = build_meta_with_fixed(B, bits=4, fixed_idx=..., fixed_val=...)`. 4. `out = gauss_vs_to_ising_states(processed, B, meta)`. 5. Optionally run `run_cim(B, meta)` to get improved candidates.

**C) Direct CIM search on QUBO encoding**
1. From `meta`, call `build_qubo_full(meta)` (provided in local `singlequbo` module). 2. Use `run_cim` which wraps the whole pipeline.

---

## 6) Performance & safety notes
- `brute_force_ising` scales as `2^N` memory/time — use `max_enumeration_bits` to avoid accidental explosions.
- The Gauss-sieve implementations are educational/small-scale: for production-level large sieving use optimized libraries.
- `v_to_u_from_B` uses `np.linalg.solve` and floating rounding — ensure `B` is well-conditioned or use exact/integer linear algebra for very large integers.


## 7) License & author
MIT license. Author: Mahmood Hasani
