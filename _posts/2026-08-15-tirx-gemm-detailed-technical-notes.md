---
layout: post
title: "TIRx GEMM - Detailed Technical Notes"
date: 2026-08-15 22:51:00 +0200
description: "Detailed technical notes for the nine-stage TIRx GEMM optimization journey on NVIDIA Blackwell."
tags: cuda gpu tirx blackwell gemm performance
categories: technical-notes
giscus_comments: true
related_posts: true
toc:
  sidebar: left
---

We start with a basic functioning version. 

# V1

## Main task

This is a deliberately simple, correctness-first Blackwell GEMM kernel. It computes:

$$
\begin{aligned}
D &= AB^T, \\
A &\in \mathbb{R}^{128 \times 64}, \\
B &\in \mathbb{R}^{128 \times 64}, \\
D &\in \mathbb{R}^{128 \times 128}.
\end{aligned}
$$

Equivalently, each output element is:

$$
D_{m,n} = \sum_{k=0}^{63} A_{m,k} B_{n,k}.
$$

The full data path is:

```text
A/B in GMEM
    ↓ cooperative synchronous copy
A/B tiles in swizzled SMEM
    ↓ tcgen05.mma
FP32 accumulator in TMEM
    ↓ tcgen05.ld
FP32 values in registers
    ↓ cast
FP16 values in registers
    ↓ ordinary stores
D in GMEM
```

One CTA containing one warpgroup, or 128 threads, computes the entire output tile. This version intentionally has no K-loop, no TMA, no pipeline, and no overlap between data movement and computation. It establishes the complete Blackwell data path before adding optimization complexity.

# V2

## K-Loop Accumulation

Version 2 extends the kernel along the contraction dimension $K$. The output remains a single $128 \times 128$ tile, computed by one CTA and one warpgroup.

The main changes are:

- **K-loop:** Split a larger $K$ into chunks of `BLK_K=64` and process them sequentially.
- **Accumulator reuse:** Reuse the same TMEM accumulator for all K chunks.
- **Accumulation control:** Use `accum=False` for the first chunk to initialize the accumulator, then `accum=True` for subsequent chunks.
- **SMEM reuse:** Load each new A/B chunk into the same pair of shared-memory buffers.
- **Barrier phase tracking:** Toggle `phase_mma` after every MMA completion because the same `mbarrier` is reused across iterations.

```text
for i in T.serial(K_TILES):
    Tx.cta.copy(Asmem[:, :], A[:, i * BLK_K : (i + 1) * BLK_K])
    Tx.cta.copy(Bsmem[:, :], B[:, i * BLK_K : (i + 1) * BLK_K])
    T.cuda.cta_sync()

    Tx.gemm_async(
        tmem[:, :BLK_N],
        Asmem[:, :],
        Bsmem[:, :],
        accum=(i != 0),
        dispatch="tcgen05",
        cta_group=1,
    )

    T.ptx.mbarrier.try_wait(mma_bar.ptr_to([0]), phase_mma)
    phase_mma ^= 1
```

Because  `try_wait(bar, phase)` blocks until the barrier’s internal phase *differs* from the `phase` argument. So the argument we pass has to name the phase we expect to leave behind, not the one we are waiting to reach. If we do not update `phase_mma` , then next wait might be returned earlier due to the previous round completion. This might produce slient errors. The execution is still sequential:

```text
load → synchronize → MMA → wait → next K chunk
```

There is no TMA, double buffering, load-compute overlap, or warp specialization yet. Step 2 adds correct full-K reduction, not asynchronous pipelining.

# V3

## Spatial Tiling with Multiple CTAs

Version 3 extends the kernel across the output dimensions $M$ and $N$. The output matrix is divided into $128 \times 128$ tiles, with one CTA assigned to each tile.

### Main Changes

- **2D CTA grid:** Launch one CTA per output tile.

```text
bx, by = T.cta_id([
    M // BLK_M,
    N // BLK_N,
])
```

This is conceptually equivalent to cuda :

```cpp
dim3 grid(M / 128, N / 128);
dim3 block(128);

kernel<<<grid, block>>>(A, B, D);
```

- **Per-CTA offsets:** Each CTA determines its output region from its grid coordinates.

```text
m_st = bx * BLK_M
n_st = by * BLK_N
```

CTA `(bx, by)` owns:

```text
D[m_st:m_st + BLK_M,
  n_st:n_st + BLK_N]
```

- **Offset operand loads:** Each CTA loads the corresponding rows of A and B during its K-loop.

```text
A[m_st:m_st + BLK_M, k:k + BLK_K]
B[n_st:n_st + BLK_N, k:k + BLK_K]
```

- **Offset writeback:** The result is written to the output tile owned by the CTA.

```text
D[m_thr, n_st:n_st + BLK_N]
```

### Execution Structure

For `M=N=256` and `BLK_M=BLK_N=128`, the launch creates a `2×2` grid:

```text
CTA (0,0) → D[0:128,   0:128]
CTA (0,1) → D[0:128, 128:256]
CTA (1,0) → D[128:256, 0:128]
CTA (1,1) → D[128:256,128:256]
```

The SMEM, TMEM, register layouts, K-loop, and `tcgen05` dispatch remain unchanged. Step 3 adds spatial parallelism across output tiles, but it still does not use TMA, pipelining, or load-compute overlap. 

# V4

# TMA Async Load and Store

Version 4 replaces thread-driven GMEM–SMEM copies with Tensor Memory Accelerator (TMA) transfers. The tile shapes, layouts, K-loop, and MMA computation remain unchanged.

Before version 4, all 128 CTA threads cooperatively copied A and B:

```text
Tx.cta.copy(Asmem, A[...])
Tx.cta.copy(Bsmem, B[...])
T.cuda.cta_sync()
```

Version 4 uses one thread to launch TMA:

```text
if tid == 0:
    Tx.copy_async(Asmem, A[...], dispatch="tma")
    Tx.copy_async(Bsmem, B[...], dispatch="tma")
```

The TMA engine handles address generation, coalescing, swizzling, and data movement.

A dedicated mbarrier tracks the number of bytes transferred:

```c
T.ptx.mbarrier.arrive.expect_tx(tma_bar, total_bytes)
T.ptx.mbarrier.try_wait(tma_bar, phase_tma)
```

The load and MMA are still sequential: `TMA load → wait → MMA → wait → next K tile`. There is no load-compute overlap yet.

# V5

# Software Pipeline with Double-Buffered SMEM

Version 5 introduces a two-stage software pipeline using `PIPE_DEPTH = 2`. Its main purpose is to remove the shared-memory storage conflict that prevented the kernel from preparing a future K tile while the current tile was being consumed.

**The Problem in version 4**

V4 uses only one pair of shared-memory buffers `Asmem[128, 64], Bsmem[128, 64]` 

The next TMA load cannot overwrite these buffers until the current MMA has finished reading them:

```text
Load k0 -> MMA k0 -> Load k1 -> MMA k1
```

Although TMA and Tensor Cores are separate hardware units, a single SMEM slot forces them to use the same storage sequentially.

**Double Buffering**

V5 adds a pipeline-stage dimension:

```text
Asmem[PIPE_DEPTH, BLK_M, BLK_K]
Bsmem[PIPE_DEPTH, BLK_N, BLK_K]
```

With `PIPE_DEPTH = 2`, the two stages form a ring buffer:

```text
Stage 0: k0 -> k2 -> k4 -> ...
Stage 1: k1 -> k3 -> k5 -> ...
```

The stage for K tile `k` is selected by:

```text
stage = k % PIPE_DEPTH
```

Before the main loop starts, the first two K tiles are prefetched:

```text
for s in range(min(PIPE_DEPTH, K_TILES)):
    tma_load(s, s * BLK_K)
```

This is the pipeline prologue. It ensures that the first MMA iteration already has data waiting in SMEM.

## Main-Loop Schedule

For each K tile, the kernel:

1. Selects its SMEM stage.
2. Waits for TMA to finish filling that stage.
3. Runs MMA using the tile stored there.
4. Waits until MMA has finished reading the stage.
5. Reuses the stage to prefetch tile `k + PIPE_DEPTH`.

Conceptually:

```text
stage = k % PIPE_DEPTH

wait_for_tma(stage)
mma(stage, accum=(k != 0))
wait_for_mma()

next_k = k + PIPE_DEPTH
if next_k < K_TILES:
    tma_load(stage, next_k * BLK_K)
```

For `K_TILES = 5`, the schedule is:

| Iteration | Stage | MMA consumes | Stage is refilled with |
| --- | --- | --- | --- |
| 0 | 0 | `k0` | `k2` |
| 1 | 1 | `k1` | `k3` |
| 2 | 0 | `k2` | `k4` |
| 3 | 1 | `k3` | Nothing |
| 4 | 0 | `k4` | Nothing |

No new prefetch is needed during the final two iterations because no future K tiles remain.

**Per-Stage TMA Barriers**

Each SMEM stage has its own TMA completion barrier:

```text
tma_bar = pool.alloc((PIPE_DEPTH,), "uint64", align=8)
```

A TMA load into stage `s` reports its completion through:

```text
tma_bar[s]
```

This allows the kernel to distinguish between:

```text
“Stage 0 is ready”
“Stage 1 is ready”
```

The MMA path waits only on the barrier corresponding to the stage it is about to consume.

**Barrier Phase Tracking**

The MMA result uses one TMEM accumulator and one `mma_bar`, which is reused every iteration. Therefore, its phase flips after every MMA:

```text
phase_mma ^= 1
```

The TMA path has one barrier per pipeline stage. A particular barrier is reused only after the stage index wraps around the ring. Therefore, `phase_tma` flips once per complete pass through the stages:

```text
if stage == PIPE_DEPTH - 1:
    phase_tma ^= 1
```

For `PIPE_DEPTH = 2`:

```text
k0: stage 0, TMA phase 0
k1: stage 1, TMA phase 0
k2: stage 0, TMA phase 1
k3: stage 1, TMA phase 1
```

Without correct phase tracking, a wait could mistake the previous use of a barrier for completion of the current transfer and release too early.

# V6

**Persistent Kernel and Tile Scheduler**

Version 6 changes how output tiles are assigned to CTAs. The per-tile computation from V5 remains largely unchanged. Step 5 is One CTA per Output Tile. For `M = N = 4096, BLK_M = BLK_N = 128`, the output contains `32 × 32 = 1024 tiles` .  What we did in Step 5 is to create a `32 × 32` CTA grid:

```text
bx, by = T.cta_id([M // BLK_M, N // BLK_N])
```

This is equivalent to 

```cpp
dim3 grid(32, 32);
kernel<<<grid, 128>>>(...);
```

Each CTA computes exactly one `128 × 128` output tile and then exits.

The 1024 CTAs are logical grid members, not concurrently resident CTAs. The GPU executes them in waves according to the available SMs and kernel resource usage.

V6 sizes the grid according to the hardware:

```text
SM_COUNT = 148
bx = T.cta_id([SM_COUNT])
```

On the target B200, this launches 148 persistent CTAs, roughly one per SM. Each CTA repeatedly asks the tile scheduler for more work:

```text
while tile_scheduler.valid():
    m_st = tile_scheduler.m_idx * BLK_M
    n_st = tile_scheduler.n_idx * BLK_N

    compute_output_tile(m_st, n_st)

    tile_scheduler.next_tile()
```

For 1024 output tiles:

```text
1024 / 148 ≈ 6.9 tiles per persistent CTA
```

**Why Persistence Helps**

1. Amortized Setup

TMEM allocation, barrier initialization, and scheduler setup happen once per CTA and are reused across several output tiles.

```text
V5:
setup -> one tile -> exit

V6:
setup -> tile 0 -> tile 1 -> ... -> exit
```

1. Better L2 Locality

The scheduler groups nearby tiles using:

```text
l2_group_size = 8
```

Tiles in the same row band reuse A tiles, while tiles in the same column band reuse B tiles. Processing them close together increases the chance that operands remain in L2 instead of being fetched again from HBM.

This is cache reuse, not explicit sharing through SMEM.

**Per-Tile State**

Each output tile starts a new K-loop, so its software phase tracking is reset inside the scheduler loop:

```text
phase_tma = 0
phase_mma = 0
```

This prevents the new tile from inheriting stale phase state from the previous tile.

V6 wraps the V5 per-tile pipeline inside a persistent outer scheduling loop, reducing repeated setup and improving L2 operand locality. 

# V7

**Warp Specialization and Pipelining**

V7 converts the single-control-path kernel from V6 into a warp-specialized pipeline. TMA load, Tensor Core computation, and writeback are assigned to different warps so the corresponding hardware paths can progress concurrently.

Step 6 already has:

- TMA asynchronous loads
- Double-buffered SMEM
- Persistent CTAs
- Tile scheduling

However, one warpgroup still controls the complete sequence `load -> wait -> MMA -> wait -> writeback `. Step 7 separates these operations into independent producer and consumer roles.

**Warp Roles**

The kernel uses two warpgroups:

| Actor | Location | Job |
| --- | --- | --- |
| TMA producer | WG 1, warp 3 | Continuously loads A and B tiles |
| MMA consumer | WG 1, warp 0 | Issues MMA when operands are ready |
| Writeback | WG 0, all four warps | Reads TMEM and stores results to GMEM |

The steady-state schedule becomes:

```text
TMA:       load k1       load k2       load k3
MMA:  compute k0    compute k1    compute k2
Store:          write tile i-1
```

Some warps remain unused, but the objective is to keep TMA, Tensor Cores, and the writeback path busy, rather than maximize CUDA thread utilization.

**SMEM Ring Buffer**

With `PIPE_DEPTH=2`, the operand buffers form a ring:

```text
Stage 0: k0 -> k2 -> k4
Stage 1: k1 -> k3 -> k5
```

Each stage moves through the following lifecycle:

```text
empty -> TMA loading -> ready -> MMA reading -> empty
```

TMA cannot overwrite a stage until MMA has finished reading it. MMA cannot read a stage until TMA has completed the corresponding transfer.

**Four Barrier Handoffs**

| Barrier | Direction | Meaning |
| --- | --- | --- |
| `tma2mma` | TMA to MMA | The SMEM operands are ready |
| `mma2tma` | MMA to TMA | The SMEM stage can be reused |
| `mma2ld` | MMA to writeback | The TMEM accumulator is ready |
| `ld2mma` | Writeback to MMA | TMEM is free for the next output tile |

The ownership cycles are:

```text
SMEM:
mma2tma -> TMA writes -> tma2mma -> MMA reads -> mma2tma

TMEM:
ld2mma -> MMA writes -> mma2ld -> writeback reads -> ld2mma
```

The barrier type matches its producer:

- `TMABar` tracks TMA completion using transaction byte counts.
- `TCGen05Bar` receives completion notifications from `tcgen05`.
- `MBarrier` counts arrivals from the writeback threads.

**PipelineState**

`PipelineState` combines the current stage and phase:

```text
tma_ps = PipelineState(PIPE_DEPTH, phase=1)
mma_ps = PipelineState(PIPE_DEPTH, phase=0)
```

The producer starts at phase 1 so its first empty-buffer wait passes immediately. The consumer starts at phase 0 so it initially blocks until the first TMA load completes.

```text
pipeline_state.advance()
```

advances to the next ring-buffer stage and updates the phase when the stage index wraps around.

Incorrect initial phases or stage advancement can cause premature execution or deadlock.

**Producer Flow**

For every K tile, the TMA producer:

```text
wait for mma2tma
load A and B into the current SMEM stage
register completion through tma2mma
advance the pipeline state
```

The first wait prevents TMA from overwriting operands still being consumed by MMA.

**MMA Consumer Flow**

For each output tile, the MMA consumer:

```text
wait for ld2mma
for each K tile:
    wait for tma2mma
    issue tcgen05 MMA
    signal mma2tma on completion
signal mma2ld after the K-loop
```

The first K tile initializes the accumulator with `accum=False`. Later K tiles use `accum=True`.

**Writeback Flow**

The writeback warpgroup performs:

```text
wait for mma2ld
TMEM -> registers
wait for tcgen05.ld
signal ld2mma
cast fp32 -> fp16
registers -> Dsmem
TMA store Dsmem -> GMEM
```

`ld2mma` can be signaled after the accumulator reaches registers because the remaining cast and store operations no longer access TMEM.

**Warpgroup Synchronization**

Only WG 0 participates in writeback, so a CTA-wide synchronization would deadlock. Step 7 instead uses:

```text
T.cuda.warpgroup_sync(10)
```

which lowers approximately to:

```text
bar.sync 10, 128
```

Here, `10` is a named barrier ID, and the synchronization waits for the 128 writeback threads.

**Pipeline Depth Cost**

One A/B stage requires:

$$
(128 \times 64 + 128 \times 64) \times 2 = 32\ \text{KiB}.
$$

The single `Dsmem` buffer requires another 32 KiB. Therefore:

$$
\text{SMEM} \approx (\text{PIPE\_DEPTH} + 1) \times 32\ \text{KiB}.
$$

Examples:

| Pipeline depth | Approximate SMEM |
| --- | --- |
| 2 | 96 KiB |
| 4 | 160 KiB |
| 6 | 224 KiB |

A deeper pipeline can hide more latency but consumes more SMEM and may reduce occupancy or exceed the hardware limit. Full kernel:

# V8

**Two-CTA Cluster**

Version 8 extends warp-specialized GEMM from one CTA to a cluster of two cooperating CTAs. The two CTAs stage different operand slices and jointly execute one larger Tensor Core MMA. From V7: 1 CTA  -\> 128 x 128 output tile to  V8: 2 CTAs -\> 256 x 256 output tile

**Cluster Structure**

```python
CTA_GROUP = 2
cbx, cby = T.cta_id_in_cluster([CTA_GROUP, 1])
```

Here the cluster shape is `[CTA_GROUP, 1] = [2, 1]` , and `cbx` identifies the CTA within the cluster: `cbx` = 0: CTA 0; `cbx` = 1: CTA 1, and `cby` is always 0.

Each CTA still owns separate SMEM and TMEM. Cluster support allows cooperative operations and remote SMEM access, but it does not merge them into one ordinary memory pool.

**Operand Partition**

For cluster tile `(m_idx, n_idx)`:

```text
m_base = m_idx * 256
n_base = n_idx * 256

m_st = m_base + cbx * 128
n_st = n_base + cbx * 128
```

The two CTAs load:

| CTA | A loaded | Stored B loaded |
| --- | --- | --- |
| CTA 0 | `A[m_base:m_base+128, :]` | `B[n_base:n_base+128, :]` |
| CTA 1 | `A[m_base+128:m_base+256, :]` | `B[n_base+128:n_base+256, :]` |

Since the operation is $D=AB^T$, rows of stored $B$ become output columns after transposition.

**Cooperative MMA**

Only one selected thread in CTA 0 issues the operation:

```text
if cbx == 0:
    Tx.gemm_async(..., cta_group=2)
```

`cta_group=2` tells the hardware to use the SMEM and TMEM resources of the CTA pair.

The cooperative MMA computes four quadrants:

$$
D_{\text{cluster}}=
\begin{bmatrix}
A_0B_0^T & A_0B_1^T\\
A_1B_0^T & A_1B_1^T
\end{bmatrix}
$$

The instruction is issued once, but the hardware reads operands from both CTAs and distributes the accumulator across their TMEM spaces.

**TMEM Result Placement**

There is no single ordinary TMEM buffer shared by both CTAs. Each CTA receives its own 128-row part:

```text
CTA 0 TMEM:
[A0 @ B0.T | A0 @ B1.T]
shape = 128 x 256

CTA 1 TMEM:
[A1 @ B0.T | A1 @ B1.T]
shape = 128 x 256
```

Together, these two local TMEM regions represent one logical `256 x 256` accumulator.

**Cross-CTA SMEM Access**

An explicit remote view can be written conceptually as:

```text
B_remote = T.ptx.map_shared_rank(Bsmem, cta_id=1)
```

From CTA 0's perspective:

```text
cta_id=0: local CTA 0 SMEM
cta_id=1: peer CTA 1 SMEM
```

`B_remote` is an address alias to CTA 1's SMEM, not a copy of B.

In the current TIRx kernel, this remote access is implicit in:

```text
Tx.gemm_async(..., cta_group=2)
```

The lowering and hardware use the two CTAs' operand descriptors to perform the cross-CTA reads.

**Cross-CTA TMA Barrier**

Both CTAs load their local A and B slices, but their TMA completion is collected in CTA 0's barrier:

```text
tma2mma_cta0 = tma2mma.remote_view(0)
```

The expected transfer size per stage is:

$$
\begin{aligned}
2\left(128 \times 64 + 128 \times 64\right)\times 2
&= 65{,}536\ \text{bytes} \\
&= 64\ \text{KiB}.
\end{aligned}
$$

MMA starts only after both CTAs' operand transfers have completed.

**`cta_id`****, ****`cta_group`****, and ****`cta_mask`**

| Parameter | Meaning |
| --- | --- |
| `cta_id=1` | Access one specific CTA's remote SMEM or barrier |
| `cta_group=2` | Execute cooperatively across a two-CTA pair |
| `cta_mask=3` | Send completion notifications to both CTAs |

`cta_mask=3` is binary `11`:

```text
bit 0 -> notify CTA 0
bit 1 -> notify CTA 1
```

It is used for:

```text
mma2tma.arrive(..., cta_group=2, cta_mask=3)
mma2ld.arrive(..., cta_group=2, cta_mask=3)
```

Both producers must learn when their SMEM stage is reusable, and both write back warpgroups must learn when their local TMEM result is ready.

**Output Ownership**

`cbx` has two different roles in the load addresses:

```text
m_st: selects the A rows and output rows owned by the CTA
n_st: selects only the stored-B slice loaded by the CTA
```

Both CTAs write all 256 columns, but only for their own 128 rows:

```text
CTA 0 writes D[m_base:m_base+128, n_base:n_base+256]
CTA 1 writes D[m_base+128:m_base+256, n_base:n_base+256]
```

Thus, each CTA writes one `128 x 256` half of the cluster output.

**Epilogue Column Address**

Writeback uses the cluster column origin rather than the per-CTA B-load offset:

```text
for no in range(2):
    n_st_epi = n_idx * 256 + no * 128
```

The two chunks are:

```text
no=0: columns n_base:n_base+128
no=1: columns n_base+128:n_base+256
```

There is no `cbx` in `n_st_epi`, because `cbx` only partitions the B-loading responsibility. It does not partition output-column ownership.

**Writeback Path**

Each CTA processes its local TMEM accumulator in two 128-column chunks:

```text
TMEM -> registers -> fp16 conversion -> Dsmem -> TMA store -> GMEM
```

Chunking avoids keeping all 256 FP32 output values per thread live simultaneously.

After both CTAs finish using cooperative TMEM, the barrier requires $128 \times 2 = 256$ thread arrivals.

A final `cluster_sync()` ensures both CTAs are finished before cooperative TMEM deallocation. Full kernel: 

# V9

**Multi-Consumer Warp Specialization**

V9 extends the two-CTA cooperative kernel from V8 by adding a second MMA consumer. The two consumers process different A rows while reusing the same staged B tile.

```text
Step 8: 1 consumer  -> 256 x 256 cluster output
Step 9: 2 consumers -> 512 x 256 cluster output
```

Here, a consumer is a warp responsible for issuing and coordinating an asynchronous MMA.

**Why Share B?**

The GEMM is $D=AB^T$. The two consumers compute different output rows but the same output columns:

```text
Consumer 0: A0 x B.T
Consumer 1: A1 x B.T
```

Different output rows require different A blocks. The same output columns require the same B rows, making B the natural operand to share.

The complete cluster result is:

$$
D_{\text{cluster}}=
\begin{bmatrix}
A_0B^T\\
A_1B^T
\end{bmatrix}
$$

**Work Partition Across the CTA Pair**

Each consumer still issues one `cta_group=2` cooperative MMA.

| Consumer | CTA 0 A rows | CTA 1 A rows | Shared B rows | Output |
| --- | --- | --- | --- | --- |
| 0 | first 128 | next 128 | 256 rows | first 256 output rows |
| 1 | next 128 | next 128 | same 256 rows | second 256 output rows |

Together, the consumers cover a `512 x 256` output tile.

**SMEM Layout**

V8 stores one A block per stage `Asmem[PIPE_DEPTH, BLK_M, BLK_K]. `V9 adds a consumer dimension`Asmem[PIPE_DEPTH, NUM_CONSUMER, BLK_M, BLK_K]` . 

The two A blocks are:

```text
Asmem[stage, 0] -> consumer 0
Asmem[stage, 1] -> consumer 1
```

B remains shared `Bsmem[PIPE_DEPTH, BLK_N, BLK_K]` .  

**TMA Load Volume**

Each CTA loads two A blocks and one B block per stage. Across two CTAs:

$$
\begin{aligned}
2\left(2 \times 128 \times 64 + 128 \times 64\right)\times 2
&= 98{,}304\ \text{bytes} \\
&= 96\ \text{KiB}.
\end{aligned}
$$

The B load volume remains unchanged from V8, but each B tile participates in two cooperative MMAs.

**Warp Roles**

V9 uses three warpgroups per CTA:

| Location | Role |
| --- | --- |
| WG 2, warp 3 | TMA producer |
| WG 2, warp 0 | MMA consumer 0 |
| WG 2, warp 1 | MMA consumer 1 |
| WG 0, all warps | Writeback for consumer 0 |
| WG 1, all warps | Writeback for consumer 1 |
| WG 2, warp 2 | Unused |

Within each MMA consumer warp, one thread selected by `elect_sync()` issues the cooperative MMA.

**TMEM Partition**

The two consumers write into separate TMEM column ranges:

```text
Consumer 0 -> TCols [0:256]
Consumer 1 -> TCols [256:512]
```

Conceptually:

```text
TMEM [0:256]   = accumulator for A0 x B.T
TMEM [256:512] = accumulator for A1 x B.T
```

This prevents the two asynchronous MMA streams from overwriting each other's accumulators.

**Barrier Changes**

TMA to MMA

`tma2mma` becomes ready only after all two-A-plus-one-B transfers for the stage have completed. Both consumers then read the same ready stage.

MMA to TMA

The producer may reuse a stage only after both consumers have finished reading it:

```text
mma2tma.init(NUM_CONSUMER)  # two arrivals
```

One arrival is insufficient because the other consumer may still be using B or its A block.

MMA to Writeback

`mma2ld` has one slot per consumer:

```text
slot 0 -> consumer 0 result
slot 1 -> consumer 1 result
```

Writeback to MMA

`ld2mma` also has one slot per consumer. Each slot prevents its corresponding TMEM range from being overwritten before writeback has loaded the accumulator into registers.

The MMA side selects slots using `warp_id`, while the writeback side selects the same slots using `wg_id`.

**Writeback**

Each writeback warpgroup handles one consumer:

```text
WG 0 reads TMEM [0:256]
WG 1 reads TMEM [256:512]
```

The 256 columns are processed in four chunks:

```text
EPI_N = 64
256 / 64 = 4
```

Chunking limits the number of live FP32 registers per thread.

The writeback warpgroups use separate named barriers:

```text
WG 0 -> barrier ID 10
WG 1 -> barrier ID 11
```

This prevents their synchronization arrivals from being combined accidentally.

**Tile Scheduler**

The persistent scheduler now assigns `512 x 256` cluster tiles:

```text
num_m_tiles = M // 512
num_n_tiles = N // 256
```

The second consumer expands the tile along M while preserving the same N range and B data.

