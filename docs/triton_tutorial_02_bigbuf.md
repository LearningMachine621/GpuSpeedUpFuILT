# Triton Tutorial 02 — bigbuf fuse4 / split4

**Goal:** Port `fuse4_from_bigbuf_kernel` and `split4_to_bigbuf_kernel` (CUDA) → Triton.
**Target file:** `src/fuilt/fusion_split/triton_bigbuf_ext.py`
**You write:** ~6 lines (fuse4) + ~12 lines (split4) in two separate TODO blocks.
**Verifies:** `python -m fuilt.fusion_split.triton_bigbuf_ext`

This tutorial builds on [Tutorial 01](./triton_tutorial_01_pointwise.md). If you
haven't done that one yet, go back — Tutorial 02 assumes you understand
`tl.program_id`, `tl.arange`, `tl.load(mask=)`, and `tl.store(mask=)`.

The new complexity in Tutorial 02:
- **2D coordinates** (we use a 1D index that we decompose into `(x, y)`)
- **4 input/output regions** per kernel (A, B, C, D sub-blocks)
- **Arithmetic masks** (`(x < S) & (y >= P)` etc.) — each output pixel sees a
  different subset of the 4 sub-blocks
- **Scatter pattern** in split4 (1 read → 4 writes)

---

## Part 0 — The geometry

This is the single most important thing to internalize. **Read this twice.**

We have one big buffer (`in_big`, shape `[H, W]`). Inside it are 4 sub-blocks
A, B, C, D scattered at arbitrary positions, plus an output region O. Each is
identified by its top-left corner:

```
   in_big[H, W]:
   ┌─────────────────────────────────────────────┐
   │                                              │
   │     ┌─────────┐         ┌─────────┐          │
   │     │   A     │         │   B     │          │
   │  (ax0,ay0)    │    (bx0,by0)     │          │
   │     └─────────┘         └─────────┘          │
   │                                              │
   │     ┌─────────┐         ┌─────────┐          │
   │     │   C     │         │   D     │          │
   │  (cx0,cy0)    │    (dx0,dy0)     │          │
   │     └─────────┘         └─────────┘          │
   │                                              │
   │                       ┌────────────────┐     │
   │                       │      O (out)   │     │
   │                       │  (ox0, oy0)    │     │
   │                       │  S_out × S_out │     │
   │                       └────────────────┘     │
   │                                              │
   └─────────────────────────────────────────────┘
```

Each sub-block A/B/C/D has side length `S`. They are arranged with overlap `O`,
so the stride between adjacent sub-blocks is `P = S - O`. The output region is
`S_out × S_out` where `S_out = S + P`.

### The fuse4 coverage map (output-local)

The output region is logically divided into 4 quadrants. Each output pixel `(x, y)`
in output-local coords `[0, S_out) × [0, S_out)` is covered by:

```
   Output region (S_out × S_out):
   ┌──────────────┬──────────────┐
   │              │              │       x ∈ [0, S)
   │  A only      │  A + B       │   and  x ∈ [P, P+S)
   │              │              │       both with y ∈ [0, S)
   ├──────────────┼──────────────┤
   │              │              │
   │  A + C       │  A+B+C+D     │       y ∈ [P, P+S)
   │              │              │
   └──────────────┴──────────────┘
        x ∈ [0,S)      x ∈ [P, P+S)
```

- Top-left quadrant: only A covers it
- Top-right: A + B (sub-block B is shifted right by P)
- Bottom-left: A + C (sub-block C is shifted down by P)
- Bottom-right: A + B + C + D

This is why `mask_A`, `mask_B`, `mask_C`, `mask_D` are computed independently —
each output pixel sees a different subset.

### The split4 scatter map

split4 is the inverse: read **one** parent block of size `S_out × S_out` and
scatter its 4 quadrants back into 4 child sub-blocks.

For each child-local `(x, y) ∈ [0, S) × [0, S)`:

```
   child A[x, y] = parent[ix0 + x,     iy0 + y]       # top-left quadrant
   child B[x, y] = parent[ix0 + P + x, iy0 + y]       # top-right quadrant
   child C[x, y] = parent[ix0 + x,     iy0 + P + y]   # bottom-left
   child D[x, y] = parent[ix0 + P + x, iy0 + P + y]   # bottom-right
```

Each thread writes 4 output pixels.

---

## Part 1 — fuse4 kernel walkthrough

The kernel has 6 conceptual steps. Steps 1–5 are **already written** for you in
the framework. Step 6 is your TODO.

### Step 1 — 1D grid decomposition

```python
pid = tl.program_id(0)
idx = pid * BLOCK + tl.arange(0, BLOCK)   # 1D index into S_OUT*S_OUT output
```

We use a 1D grid because the output region is contiguous in `(y * S_OUT + x)`
order. Each program handles `BLOCK` output pixels.

### Step 2 — 1D → 2D decomposition

```python
x = idx % S_OUT
y = idx // S_OUT
```

`x` is the column (horizontal), `y` is the row (vertical). Both are
BLOCK-wide int32 vectors.

### Step 3 — Global output coordinates

```python
gx = ox0 + x       # global column in [0, W)
gy = oy0 + y       # global row    in [0, H)
```

### Step 4 — Output mask (boundary handling)

```python
out_mask = (idx < n_out) & (gx < W) & (gy < H)
```

True where this output pixel exists. Note `idx < n_out` handles the tail of
the grid (last program may extend past `S_OUT * S_OUT`).

### Step 5 — Coverage masks + loads

```python
mask_A = (x < S) & (y < S)
mask_B = (x >= P) & (x < P + S) & (y < S)
mask_C = (x < S) & (y >= P) & (y < P + S)
mask_D = (x >= P) & (x < P + S) & (y >= P) & (y < P + S)

a = tl.load(in_ptr + (ay0 + y) * W + (ax0 + x),         mask=mask_A, other=0.0)
b = tl.load(in_ptr + (by0 + y) * W + (bx0 + x - P),     mask=mask_B, other=0.0)
c = tl.load(in_ptr + (cy0 + y - P) * W + (cx0 + x),     mask=mask_C, other=0.0)
d = tl.load(in_ptr + (dy0 + y - P) * W + (dx0 + x - P), mask=mask_D, other=0.0)
```

After this, `a / b / c / d` are BLOCK-wide fp32 vectors. Wherever the
corresponding mask was False, the load returned `0.0` (the `other=` value).

**Note the address arithmetic** for each sub-block:
- A is at `(ay0 + y, ax0 + x)` — direct hit
- B is at `(by0 + y, bx0 + x - P)` — B is shifted right by `P` relative to A's
  position in the output, so when output `x ∈ [P, P+S)`, B's local coord is `x - P`
- C and D follow the same logic with `y - P`

### Step 6 — Mean + store  (← **you write this**)

#### What goes here

Compute the mean over covered sub-blocks, then store with cast to fp16.

#### Hints (read in order)

**Hint 1 — Count covered sub-blocks.**
You have 4 bool masks. Convert each to int32 (0/1) and sum:
```python
count = mask_A.to(tl.int32) + mask_B.to(tl.int32) + mask_C.to(tl.int32) + mask_D.to(tl.int32)
```

**Hint 2 — Sum loaded values.**
Where mask was False, `other=0.0` was returned, so masked-out sub-blocks
contribute zero to the sum:
```python
total = a + b + c + d
```

**Hint 3 — Mean with divide-by-zero guard.**
The corners of the output region are only covered by A (count == 1), so this
isn't strictly needed — but the bottom-right corner has count == 4 and
boundary pixels might have count == 0 due to global bounds. Be safe:
```python
mean = tl.where(count > 0, total / count, 0.0)
```

**Hint 4 — Store with fp16 cast.**
Match the CUDA kernel's hardcoded fp16 output:
```python
tl.store(out_ptr + gy * W + gx, mean.to(tl.float16), mask=out_mask)
```

<details>
<summary><b>Hint 5 — Full fuse4 solution (click only if stuck)</b></summary>

```python
count = mask_A.to(tl.int32) + mask_B.to(tl.int32) + mask_C.to(tl.int32) + mask_D.to(tl.int32)
total = a + b + c + d
mean  = tl.where(count > 0, total / count, 0.0)
tl.store(out_ptr + gy * W + gx, mean.to(tl.float16), mask=out_mask)
```

</details>

---

## Part 2 — split4 kernel walkthrough

split4 is structurally simpler than fuse4 (no averaging, just 4 copies) but
the per-thread work is reversed (1 read → 4 writes).

### Steps 1–5 (already written)

Same as fuse4 for steps 1–4. Step 5 reads 4 parent values:

```python
pa_x = ix0 + x;     pa_y = iy0 + y
pb_x = ix0 + P + x; pb_y = iy0 + y
pc_x = ix0 + x;     pc_y = iy0 + P + y
pd_x = ix0 + P + x; pd_y = iy0 + P + y

pa_mask = in_child & (pa_x < W) & (pa_y < H)
pb_mask = in_child & (pb_x < W) & (pb_y < H)
pc_mask = in_child & (pc_x < W) & (pc_y < H)
pd_mask = in_child & (pd_x < W) & (pd_y < H)

va = tl.load(in_ptr + pa_y * W + pa_x, mask=pa_mask, other=0.0)
vb = tl.load(in_ptr + pb_y * W + pb_x, mask=pb_mask, other=0.0)
vc = tl.load(in_ptr + pc_y * W + pc_x, mask=pc_mask, other=0.0)
vd = tl.load(in_ptr + pd_y * W + pd_x, mask=pd_mask, other=0.0)
```

After this you have 4 BLOCK-wide fp32 vectors ready to scatter.

### Step 6 — Scatter to 4 child positions  (← **you write this**)

#### Hints

For each sub-block (A, B, C, D), do three things:

**Pattern (use this for all 4):**
```python
oa_x = ax0 + x;     oa_y = ay0 + y
oa_addr = oa_y * W + oa_x
oa_mask = in_child & (oa_x < W) & (oa_y < H)
tl.store(out_ptr + oa_addr, va.to(tl.float16), mask=oa_mask)
```

Repeat for B (bx0, by0, vb), C (cx0, cy0, vc), D (dx0, dy0, vd).

<details>
<summary><b>Full split4 solution (click only if stuck)</b></summary>

```python
# A
oa_x = ax0 + x; oa_y = ay0 + y
tl.store(out_ptr + oa_y * W + oa_x, va.to(tl.float16),
         mask=in_child & (oa_x < W) & (oa_y < H))
# B
ob_x = bx0 + x; ob_y = by0 + y
tl.store(out_ptr + ob_y * W + ob_x, vb.to(tl.float16),
         mask=in_child & (ob_x < W) & (ob_y < H))
# C
oc_x = cx0 + x; oc_y = cy0 + y
tl.store(out_ptr + oc_y * W + oc_x, vc.to(tl.float16),
         mask=in_child & (oc_x < W) & (oc_y < H))
# D
od_x = dx0 + x; od_y = dy0 + y
tl.store(out_ptr + od_y * W + od_x, vd.to(tl.float16),
         mask=in_child & (od_x < W) & (od_y < H))
```

</details>

---

## Part 3 — Verify

Once both TODO blocks are filled:

```bash
python -m fuilt.fusion_split.triton_bigbuf_ext
```

You should see:

```
============================================================
bigbuf Triton port vs CUDA reference (numerical equivalence)
============================================================
  kernel            : fuse4_from_bigbuf
  region            : 1984x1984
  max_abs_diff      : 0.0005         ← want < 1e-3 (fp16 tolerance)
  mean_abs_diff     : 0.0001
  pass              : True
------------------------------------------------------------
  kernel            : split4_to_bigbuf
  max_abs_diff      : 0.0            ← exact match (just copies)
  mean_abs_diff     : 0.0
  pass              : True
------------------------------------------------------------
```

`atol=1e-3` because we cast fp32 → fp16 → fp32 in the comparison. If you see
larger diffs, double-check the address arithmetic for sub-block B/C/D (the
`- P` and `+ P` offsets).

---

## Part 4 — Why this is harder than Tutorial 01

| Concept | Tutorial 01 (pointwise) | Tutorial 02 (bigbuf) |
|---|---|---|
| Grid dimension | 1D over flat array | 1D, but decomposed to 2D via `% S_OUT` |
| Per-thread input | 1 load | 4 conditional loads (different addresses) |
| Per-thread output | 1 store | 1 (fuse4) or 4 (split4) stores |
| Masks | 1 boundary mask | 4 coverage masks + boundary mask |
| Math | pure arithmetic | arithmetic + count + divide-by-zero guard |
| Memory pattern | perfectly coalesced (contiguous offsets) | gather/scatter (offsets depend on x, y) |

The good news: after this tutorial, you've seen **all** the Triton patterns
that show up in production serving kernels (attention, paged KV cache, RoPE,
etc. all involve some flavor of scatter/gather). Tutorial 03 (if we do one)
would be about reductions — `tl.reduce`, `tl.sum` over an axis — which is the
last piece of the Triton mental model.

---

## Part 5 — What's next

After both kernels land and verify passes:

1. **Benchmark Triton vs CUDA.** The micro-bench in `verify_*_vs_cuda` is a
   good starting point; for a real pipeline-level comparison, wire the
   `triton_bigbuf` path into the v8 host (Tutorial 04 / task D).
2. **Try different `BLOCK_SIZE`.** The default 1024 processes 32×32 output
   tiles per program. Larger blocks (4096 = 64×64) reduce launch count but
   increase register pressure. `@triton.autotune` can sweep this
   automatically if you want — see `docs/triton_tutorial_01_pointwise.md`
   Part 4 for the autotune decorator pattern.
3. **Reflect on the trade-off.** bigbuf is more complex than pointwise but
   still well within Triton's wheelhouse. The CUDA originals are ~80 lines
   each; the Triton ports are ~30 lines each. Same lesson as Tutorial 01:
   you're trading low-level control for portability + readability.
