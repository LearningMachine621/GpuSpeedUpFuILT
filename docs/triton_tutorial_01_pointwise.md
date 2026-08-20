# Triton Tutorial 01 — Pointwise Kernel

**Goal:** Port `fused_pointwise_grad_loss_kernel` (CUDA) → Triton.
**Target file:** `src/fuilt/ilt/func/triton_fused_pointwise.py`
**You write:** ~5–6 lines of math (the TODO block).
**Verifies:** `python -m fuilt.ilt.func.triton_fused_pointwise`

This is a *pointwise* kernel: every output element is computed independently
from the corresponding input element. No reductions, no shared memory, no
warp-level cooperation. It's the simplest class of GPU kernel — and the right
place to build Triton muscle memory before tackling `bigbuf`'s scatter/gather.

---

## Part 0 — The math you're computing

For each element `i`:

```
z_i       = print_steepness * (aerial_i - target_density)
printed_i = sigmoid(z_i)
diff_i    = printed_i - target_i
grad_i    = 2 * diff_i * print_steepness * printed_i * (1 - printed_i)
diff_sq_i = diff_i ** 2
```

`grad_i` is `dLoss/dAerial_i` for an L2-style loss `L = sum((printed - target)^2)`.
The chain rule gives the `2 * diff * print_steepness * printed * (1 - printed)`
form — `sigmoid'`(z) = `printed * (1 - printed)`.

That's it. Five lines.

---

## Part 1 — Triton's mental model

A Triton kernel runs as a **grid of programs**. Each program is identified by
`tl.program_id(axis)` and processes a tile of the data.

```
 ┌──────────────────────────────────────────────────────────┐
 │  Grid (1D, num_programs = cdiv(N, BLOCK_SIZE))            │
 │                                                            │
 │  program 0     program 1     program 2    ...  program P-1│
 │  [0..B)       [B..2B)       [2B..3B)            [P*B..N) │
 │                                                            │
 │  Each program:                                             │
 │    1. computes its tile of offsets                         │
 │    2. loads its tile of inputs                             │
 │    3. does math                                            │
 │    4. stores its tile of outputs                           │
 └──────────────────────────────────────────────────────────┘
```

Inside one program, **threads are managed for you** by Triton's compiler
(no `threadIdx.x`, no warp shuffles, no explicit vectorization). You write
**block-level** code: load a vector, do math on a vector, store a vector.

Contrast with CUDA: you'd write `idx = blockIdx.x * blockDim.x + threadIdx.x`,
manually worry about vectorized loads, mask tail threads, etc. Triton hides
all of this behind `tl.load` / `tl.store` + `mask=`.

---

## Part 2 — The 5 steps

### Step 1 — Find your program's identity

```python
pid = tl.program_id(0)   # this program's id in axis 0
```

`0` means the first (and for us, only) grid axis. If you had a 2D grid you'd
call `tl.program_id(1)` for the second axis too.

### Step 2 — Compute which elements this program owns

```python
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
mask    = offsets < n_elements
```

`tl.arange(0, BLOCK_SIZE)` is the equivalent of `threadIdx.x` for the whole
block — a vector `[0, 1, 2, ..., BLOCK_SIZE-1]`.

`mask` is True for elements that actually exist. The last program might own
elements past the end of the array; those positions in `offsets` are
out-of-bounds and must be skipped on every load and store.

### Step 3 — Load your tile

```python
a = tl.load(aerial_ptr + offsets, mask=mask, other=0.0)
t = tl.load(target_ptr + offsets, mask=mask, other=0.0)
```

`aerial_ptr + offsets` is **pointer arithmetic on a vector of offsets** —
Triton understands this. The result is a vector of `BLOCK_SIZE` fp32 values
sitting in registers (typically — Triton may spill to shared if needed).

`other=0.0` is the fill value for masked-out positions. You won't write to
those positions anyway, so any value works; `0.0` is conventional.

`a` and `t` are now **vectors of length `BLOCK_SIZE`** in registers.

### Step 4 — Math  (← **you write this**)

#### What goes here

Compute these two vectors from `a`, `t`:

```
grad     = ...   # BLOCK_SIZE-wide fp32 vector
diff_sq  = ...   # BLOCK_SIZE-wide fp32 vector
```

#### The math, restated

```
z       = print_steepness * (a - target_density)
printed = sigmoid(z)
diff    = printed - t
grad    = 2 * diff * print_steepness * printed * (1 - printed)
diff_sq = diff * diff
```

#### Hints (read in order; stop when you don't need more)

**Hint 1 — Triton ops are vectorized.**
`a - target_density` works because Triton broadcasts the Python-side scalar
`target_density` (which is `tl.constexpr` here) across the vector. Same for
`* print_steepness`, `* 2.0`, etc.

**Hint 2 — Sigmoid.**
`tl.sigmoid(x)` is one PTX instruction on Ada/Hopper. Use it directly.
(Manual `1.0 / (1.0 + tl.exp(-x))` works too but is two ops and offers no
advantage in fp32.)

**Hint 3 — Common subexpression.**
`printed * (1 - printed)` appears once in `grad`. Stash it in a temp:
```python
sigmoid_deriv = printed * (1.0 - printed)
```
Then `grad = 2.0 * diff * print_steepness * sigmoid_deriv`.
Triton's compiler is decent but not magical about CSE; explicit is safer.

**Hint 4 — Final form.**
Four to six lines. No loops, no conditionals, no `tl.where`. Pure arithmetic.

<details>
<summary><b>Hint 5 — One valid solution (click only if stuck)</b></summary>

```python
z              = print_steepness * (a - target_density)
printed        = tl.sigmoid(z)
diff           = printed - t
sigmoid_deriv  = printed * (1.0 - printed)
grad           = 2.0 * diff * print_steepness * sigmoid_deriv
diff_sq        = diff * diff
```

</details>

### Step 5 — Store your tile

Uncomment these two lines below the TODO:

```python
tl.store(grad_ptr + offsets, grad, mask=mask)
tl.store(diff_sq_ptr + offsets, diff_sq, mask=mask)
```

The `mask=mask` is critical — without it, the last program will write past the
end of the output buffer and corrupt whatever follows in GPU memory.

---

## Part 3 — Verify

Once you've filled in Step 4 and uncommented the two stores:

```bash
python -m fuilt.ilt.func.triton_fused_pointwise
```

This runs `verify_vs_cuda()`, which:

1. Compiles the original CUDA kernel as the oracle.
2. Runs both kernels on the same 1M-element random input.
3. Compares outputs element-wise.

You should see output like:

```
============================================================
Triton port vs CUDA reference (numerical equivalence)
============================================================
  n_elements            : 1048576
  block_size            : 1024
  grad_max_abs_diff     : 1.2e-06      ← want < 1e-5
  diff_sq_max_abs_diff  : 0.8e-06      ← want < 1e-5
  grad_pass             : True
  diff_sq_pass          : True
  overall_pass          : True
============================================================
PASS
```

If you see `FAIL`: check that you uncommented both `tl.store` lines and that
`grad` and `diff_sq` are the **last two** SSA values you compute.

---

## Part 4 — Why the Triton version is "better" than CUDA

Even though both kernels compute the same thing:

| Aspect | CUDA original | Triton port |
|---|---|---|
| Lines of code | 91 | ~20 (kernel only) |
| Manual `__device__` template handling | yes | none |
| Manual `<<<grid, block>>>` launch math | yes | hidden by `[grid]` syntax |
| Vectorized load/store | manual (you'd write it) | automatic (`tl.load`) |
| Tail masking | manual (`if (idx >= N) return;`) | automatic (`mask=`) |
| `target_density` / `print_steepness` specialization | no (runtime args) | yes (`tl.constexpr` → kernel specializes per value pair) |
| Tunable `BLOCK_SIZE` with autotune | would require manual switch | `@triton.autotune` decorator (not used here, but easy to add) |
| Maintainer onboarding cost | high (CUDA + ATen + pybind11) | low (Python) |

This is the same set of trade-offs that pushed vLLM, SGLang, and Triton-based
serving stacks to migrate their hot kernels from raw CUDA to Triton. You're
not losing performance — Triton's PTX backend is competitive with hand-tuned
CUDA on elementwise and standard reduction patterns — and you gain
portability, readability, and a much larger pool of contributors.

---

## Part 5 — What's next

After this kernel lands:

1. **Run it inside v7's CUDA Graph capture.** The Triton kernel JIT-compiles
   on the first call (during warmup), and subsequent calls inside the graph
   just replay the compiled binary. No CUDA Graph-specific changes needed.
2. **Wire it into v7.** Replace `ext.fused_pointwise_forward(...)` with
   `fused_pointwise_forward_triton(...)` in
   `baseline/test_baseline_v7_fuselevelset.py:179-180`. Run the benchmark,
   compare numbers against the CUDA baseline.
3. **bigbuf port (Tutorial 02).** Same skeleton, but the math involves
   2D scatter/gather — `tl.load`/`tl.store` with computed offsets rather
   than `+ offsets`.
