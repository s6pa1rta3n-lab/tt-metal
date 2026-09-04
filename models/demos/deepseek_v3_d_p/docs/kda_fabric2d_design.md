# KDA Fabric2D Support Design

Work: `tt-metal_tracker-66t.4`
Supporting RCA: [kda_fabric2d_rca.md](kda_fabric2d_rca.md)

## Problem

KDA's TP collective contract is implicit and differs from MLA's: topology is caller-selected as a program tuning value, and SP1 causes the TP axis to be omitted. The retained SP1xTP8 hang is not reproducible, but KDA still needs a coherent Fabric2D contract and Fabric2D-default test coverage.

Goals:

- make the active fabric the sole source of collective topology;
- make the TP axis explicit for every TP reduce-scatter;
- run every multi-device KDA test on Fabric2D by default, retaining explicit torus coverage;
- keep numerical, determinism, trace, cache, and performance acceptance visible.

Non-goals: change CCL kernels, diagnose an unretained historical runtime failure, redesign KDA math, or make single-device tests open fabric.

## Current system

- **Observed:** KDA's program config owns `tp_ccl_topology`, mixing runtime fabric geometry with kernel tuning (`tt/kda/config.py` at baseline).
- **Observed:** KDA's output projection reduces on no named axis for SP1 and the TP axis otherwise (`tt/kda/kda.py` at baseline).
- **Observed:** test construction chooses Ring from SP size, not from active fabric (`tests/kda/utils.py` at baseline).
- **Observed:** MLA keeps SP/TP topology distinct and supplies `cluster_axis=self.tp_axis` to TP reduce-scatter (`tt/mla/mla.py:452-466`, `tt/mla/mla.py:1263-1273`).
- **Observed:** `per_axis_topology()` already owns the DeepSeek fabric-to-axis mapping and queries the active fabric when no value is supplied (`tt/tt_ccl.py:372-413`).
- **Observed:** multi-device KDA component, distributed, acceptance, and performance tests previously selected Fabric1D; TorusXY was already a distinct 8x4 case.

## Proposed design

At KDA construction, resolve the TP topology exactly once as `per_axis_topology()[tp_axis]`. Keep that resolved value immutable for the layer lifetime. At output projection, use `tp_axis` consistently for semaphore selection, link count, and `cluster_axis`.

```text
active fabric -> per_axis_topology() -> select tp_axis -> ttKDA.tp topology
                                                        |
output projection -> TP semaphores/links + explicit tp_axis + topology -> reduce-scatter
```

Remove topology from `KDAProgramConfig`; program config remains limited to compute/program tuning. Keep the current CCL manager during this change: replacing `models.tt_transformers.tt.ccl.TT_CCL` with the DeepSeek manager is a separate ownership/consolidation decision and is not required to establish the contract.

Use plain Fabric2D for every KDA test that opens more than one device. Keep TorusXY cases separate because they assert a different physical contract. Single-device tests remain fabric-disabled.

## Contracts

### Topology resolution

- Precondition: fabric is configured before a multi-device `ttKDA` is constructed.
- Input: active `ttnn.FabricConfig` and KDA `tp_axis` in `{0, 1}`.
- Guarantee: plain Fabric2D selects Linear on both axes; TorusX selects Ring only on axis 1; TorusY only on axis 0; TorusXY on both.
- Ambiguity: unknown non-ring fabric falls back to Linear; unknown ring/torus-named fabric follows the shared helper's warning-and-Linear behavior.
- Invariant: model/test code cannot override topology through `KDAProgramConfig`.

### TP reduce-scatter

- Precondition: TP size greater than one requires a CCL manager.
- Guarantee: semaphore pools, link count, and the operation all receive the same explicit `tp_axis`; output remains reduce-scattered on hidden width.
- Invariant: SP size does not change TP topology or axis selection.

### Test profile

- Every multi-device KDA test opens plain Fabric2D unless the case explicitly validates a torus profile.
- Torus tests retain topology/descriptor feasibility markers; a skip is reported but does not establish hardware coverage.
- Performance references are keyed/calibrated for the actual fabric contract and are not copied from Fabric1D.

## Invariants and acceptance criteria

1. SP1xTP8, SP2xTP4, and SP4xTP2 complete eager and traced Fabric2D execution without recovery events.
2. Output, recurrent state, and convolution state meet PCC 0.9995 against the independent Torch reference.
3. Repeated execution is bit-identical where the current acceptance test requires it, and program-cache entries do not grow after warm-up.
4. Plain Fabric2D resolves Linear for both TP axes; supported torus tests resolve Ring only on wrapped axes.
5. No KDA test retains Fabric1D configuration.
6. Fabric-specific performance baselines are measured and enforced separately.

## Decisions and trade-offs

| Decision | Preferred | Alternative | Trade-off |
| --- | --- | --- | --- |
| Topology owner | shared `per_axis_topology()` | KDA-local mapping or caller value | Reuse prevents another source of truth; it creates an intentional dependency on DeepSeek CCL policy. |
| Resolution time | once at layer construction | every forward | Fabric cannot safely change while a layer and its semaphores are live; construction keeps the invariant local. |
| TP axis | always explicit | omit it on 1xN | Explicit is uniform and MLA-aligned; it may expose runtime differences hidden by linearized device-list behavior. |
| CCL manager | retain current manager | migrate in same change | Smaller blast radius; manager convergence can happen separately with its own validation. |
| Test default | Fabric2D | dual Fabric1D/Fabric2D matrix | Meets the product direction without doubling scarce hardware runtime. Torus remains targeted coverage. |

## Blast radius

- KDA Python API: remove `tp_ccl_topology` from `KDAProgramConfig` and `kimi_k3_program_config()`.
- KDA layer: resolve shared per-axis topology and use explicit TP axis.
- KDA tests: update construction assertions and all multi-device fabric profiles; recalibrate perf references.
- Shared DeepSeek CCL: dependency only, no source change.
- C++/firmware: no change; the RCA provides no evidence for one.

## Risks, assumptions, and unknowns

- **Risk:** SP1xTP8 Linear was ~14% slower than the pre-prototype Ring trace in one session. Performance acceptance must reflect the supported topology, and optimization should be separate from correctness.
- **Assumed:** production configures fabric before constructing model layers, as test fixtures do.
- **Unknown:** TorusXY behavior remains unexecuted locally because the required 8x4 host was unavailable.
- **Unknown:** the historical hang may recur on its original runtime/firmware even though it does not reproduce here.

## Evidence

The throwaway prototype demonstrated:

- three eager Fabric2D layouts passing at T=128 with output PCC 0.999950–0.999954, recurrent PCC 0.999790–0.999797, convolution PCC 0.999997;
- T=5120 SP2xTP4 output PCC 0.999952, recurrent PCC 0.999765, convolution PCC 0.999997, and three-run bit identity;
- 50 trace replays per layout at medians 11.062 ms (SP1xTP8), 9.890 ms (SP2xTP4), and 10.055 ms (SP4xTP2);
- stable program-cache entry count in both TP-axis recurrence cases.

Exact commands and pre-prototype evidence are in the RCA.

## Approval gate

The task direction explicitly approved the MLA-aligned ring/line/torus model, Fabric2D as the eventual default for all KDA multi-device tests, autonomous completion through the dev spec, and a throwaway prototype. The decisions above implement that approved direction. Permanent implementation remains gated on review of this document; the accompanying dev spec is a ready implementation plan, not authorization to ship the prototype.
