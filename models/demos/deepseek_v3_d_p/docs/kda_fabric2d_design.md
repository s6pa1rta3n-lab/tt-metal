# KDA Fabric2D Support Design

Work: `tt-metal_tracker-66t.4`
Supporting RCA: [kda_fabric2d_rca.md](kda_fabric2d_rca.md)

## Problem

KDA's collective contract is implicit and differs from MLA's production path: topology is caller-selected as a program tuning value, SP collectives rely on TTNN defaults, and SP1 causes the TP axis to be omitted. The retained SP1xTP8 hang is not reproducible, but KDA still needs a coherent Fabric2D contract and Fabric2D-default test coverage.

Goals:

- derive one explicit per-physical-axis topology tuple from the configured fabric;
- require KDA to use the corresponding tuple element for every SP and TP collective;
- make the TP axis explicit for every TP reduce-scatter;
- run every multi-device KDA test on Fabric2D by default, retaining explicit torus coverage;
- keep numerical, determinism, trace, cache, and performance acceptance visible.

Non-goals: change CCL kernels, diagnose an unretained historical runtime failure, redesign KDA math, or make single-device tests open fabric.

## Current system

- **Observed:** KDA's program config owns `tp_ccl_topology`, mixing runtime fabric geometry with kernel tuning (`tt/kda/config.py` at baseline).
- **Observed:** KDA's output projection reduces on no named axis for SP1 and the TP axis otherwise (`tt/kda/kda.py` at baseline).
- **Observed:** test construction chooses Ring from SP size, not from active fabric (`tests/kda/utils.py` at baseline).
- **Observed:** MLA keeps SP/TP topology distinct and supplies `cluster_axis=self.tp_axis` to TP reduce-scatter (`tt/mla/mla.py:452-466`, `tt/mla/mla.py:1263-1273`).
- **Observed:** MLA's production runner always obtains a two-element tuple from `per_axis_topology()` and passes it through runtime construction (`tt/runners/adapters/mla.py:118-133`).
- **Observed:** MLA accepts a scalar only as backward-compatible shorthand which applies the same topology to both axes; direct cache and sparse tests exercise it by omitting the argument and falling through the `Linear` default (`tt/mla/mla.py:276-286`, `tests/cache/test_mla_cache.py:174-183`, `tests/sparse_mla/test_sparse_mla.py:405-417`).
- **Observed:** `per_axis_topology()` already owns the DeepSeek fabric-to-axis mapping and queries the active fabric when no value is supplied (`tt/tt_ccl.py:372-413`).
- **Observed:** KDA has SP-axis all-gathers in convolution carry exchange and recurrence-summary composition, and neither currently supplies topology (`tt/kda/convolution.py:43-48`, `tt/kda/recurrence.py:242-247`).
- **Observed:** multi-device KDA component, distributed, acceptance, and performance tests previously selected Fabric1D; TorusXY was already a distinct 8x4 case.

## Proposed design

The composition root derives a required topology tuple with `per_axis_topology(configured_fabric)` after choosing the fabric profile and passes it to `ttKDA`. Tuple positions describe physical mesh dimensions `(dim0, dim1)`, independently of which dimension the caller assigns to SP or TP. `ttKDA` freezes `topology[sp_axis]` and `topology[tp_axis]` for the layer lifetime.

The KDA API accepts only `tuple[ttnn.Topology, ttnn.Topology]`: no scalar shorthand and no default. A layer containing collectives on two independently wrapped axes must retain both facts. Scalar topology is used only at the leaf CCL boundary after KDA selects the appropriate tuple element.

```text
configured fabric -> per_axis_topology() -> (dim0 topology, dim1 topology)
                                                |
                                                v
                ttKDA(topology=tuple, sp_axis, tp_axis)
                     |                         |
              select sp_axis             select tp_axis
                     |                         |
       SP carry all-gathers       TP output reduce-scatter
```

Remove topology from `KDAProgramConfig`; program config remains limited to compute/program tuning. Passing the tuple as a construction dependency keeps fabric selection outside the layer and avoids an implicit query of mutable global fabric state. Keep the current CCL manager during this change: replacing `models.tt_transformers.tt.ccl.TT_CCL` with the DeepSeek manager is a separate ownership/consolidation decision and is not required to establish the contract.

Use plain Fabric2D for every KDA test that opens more than one device. Keep TorusXY cases separate because they assert a different physical contract. Single-device tests remain fabric-disabled.

## Contracts

### Topology resolution

- Precondition: the caller derives topology from the same fabric configuration used to open the mesh.
- Input: exactly two `ttnn.Topology` values ordered by physical mesh dimension, plus distinct KDA `sp_axis` and `tp_axis` values in `{0, 1}`.
- Guarantee: KDA selects SP topology with `topology[sp_axis]` and TP topology with `topology[tp_axis]`; swapping logical axes does not reinterpret tuple order.
- Guarantee: the shared resolver maps plain Fabric2D to `(Linear, Linear)`, TorusX to `(Linear, Ring)`, TorusY to `(Ring, Linear)`, and TorusXY to `(Ring, Ring)`.
- Failure: a scalar, non-tuple, or tuple of the wrong length is rejected at construction instead of being broadcast across axes.
- Ambiguity: unknown fabric behavior belongs to `per_axis_topology()`; KDA does not query fabric or invent a fallback.
- Invariant: model/test code cannot override topology through `KDAProgramConfig`.
- Invariant: topology cannot change over the lifetime of a constructed layer.

### SP collectives

- Precondition: SP size greater than one invokes convolution-carry and recurrence-summary exchange on `sp_axis`.
- Guarantee: each SP all-gather receives both explicit `cluster_axis=sp_axis` and scalar `topology=topology[sp_axis]`.
- Invariant: TP size and topology do not change SP topology selection.

### TP reduce-scatter

- Precondition: TP size greater than one requires a CCL manager.
- Guarantee: semaphore pools, link count, and the operation all receive the same explicit `tp_axis`, while the operation receives scalar `topology=topology[tp_axis]`; output remains reduce-scattered on hidden width.
- Invariant: SP size does not change TP topology or axis selection.

### Test profile

- Every multi-device KDA test opens plain Fabric2D unless the case explicitly validates a torus profile.
- Torus tests retain topology/descriptor feasibility markers; a skip is reported but does not establish hardware coverage.
- Performance references are keyed/calibrated for the actual fabric contract and are not copied from Fabric1D.

## Invariants and acceptance criteria

1. SP1xTP8, SP2xTP4, and SP4xTP2 complete eager and traced Fabric2D execution without recovery events.
2. Output, recurrent state, and convolution state meet PCC 0.9995 against the independent Torch reference.
3. Repeated execution is bit-identical where the current acceptance test requires it, and program-cache entries do not grow after warm-up.
4. Every multi-device KDA CCL names its axis and uses the topology tuple element for that physical dimension.
5. Plain Fabric2D resolves Linear for both axes; supported torus tests resolve Ring only on wrapped axes.
6. No KDA test retains Fabric1D configuration.
7. Fabric-specific performance baselines are measured and enforced separately.

## Decisions and trade-offs

| Decision | Preferred | Alternative | Trade-off |
| --- | --- | --- | --- |
| Topology mapping owner | shared `per_axis_topology()` | KDA-local mapping | Reuse prevents another source of truth; it creates an intentional dependency on DeepSeek CCL policy. |
| Topology delivery | required two-element constructor tuple | query active fabric inside KDA | Explicit dependency is testable and prevents hidden global-state coupling. |
| Accepted API shape | tuple only | MLA-style scalar-or-tuple union | KDA uses both axes. Scalar adds ambiguity solely to preserve legacy/default callers, which KDA does not have to support. |
| Resolution time | caller resolves once before layer construction | every forward | Fabric cannot safely change while a layer and its semaphores are live; construction keeps the invariant local. |
| TP axis | always explicit | omit it on 1xN | Explicit is uniform and MLA-aligned; it may expose runtime differences hidden by linearized device-list behavior. |
| CCL manager | retain current manager | migrate in same change | Smaller blast radius; manager convergence can happen separately with its own validation. |
| Test default | Fabric2D | dual Fabric1D/Fabric2D matrix | Meets the product direction without doubling scarce hardware runtime. Torus remains targeted coverage. |

## Blast radius

- KDA Python API: remove `tp_ccl_topology` from `KDAProgramConfig` and `kimi_k3_program_config()`.
- KDA layer and SP helpers: accept the tuple, select topology by logical axis, and make topology explicit on every CCL.
- KDA construction sites/tests: derive and pass the tuple, update contract assertions and all multi-device fabric profiles, and recalibrate perf references.
- Shared DeepSeek CCL: dependency only, no source change.
- C++/firmware: no change; the RCA provides no evidence for one.

## Risks, assumptions, and unknowns

- **Risk:** SP1xTP8 Linear was ~14% slower than the pre-prototype Ring trace in one session. Performance acceptance must reflect the supported topology, and optimization should be separate from correctness.
- **Assumed:** production construction has access to the fabric configuration used to open its mesh and can pass the derived tuple to KDA.
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
