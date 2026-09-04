# KDA Fabric2D Support Development Spec

Work: `tt-metal_tracker-66t.5`
Approved direction: [kda_fabric2d_design.md](kda_fabric2d_design.md)
Evidence: [kda_fabric2d_rca.md](kda_fabric2d_rca.md)

## Goal

Implement the reviewed KDA contract in which callers derive and pass a required per-physical-axis topology tuple, every KDA collective selects topology by its explicit axis, and all multi-device KDA tests default to Fabric2D. Do not carry the diagnostic probe or one-session performance numbers into the production patch.

Out of scope: C++/firmware changes, CCL-manager consolidation, Fabric1D comparison coverage, and investigation of the unretained historical failure.

## Design constraints

- **Required:** `per_axis_topology()` is the only fabric-to-topology mapping used by KDA callers.
- **Required:** `ttKDA` takes `topology: tuple[ttnn.Topology, ttnn.Topology]` as a required construction dependency. It accepts neither a scalar nor a default.
- **Required:** tuple positions always mean physical mesh dimensions `(dim0, dim1)`; logical SP/TP axes select their element by index.
- **Required:** topology cannot be supplied through program tuning or queried from global fabric state inside KDA.
- **Required:** semaphore selection, link selection, and reduce-scatter use the same TP axis.
- **Required:** convolution and recurrence all-gathers use the explicit SP axis and its selected scalar topology.
- **Required:** plain Fabric2D is the default for multi-device KDA tests; torus is explicit; single-device tests need no fabric.
- **Required:** numerical, state, determinism, cache, trace, and fabric-specific performance contracts remain gated.

## Existing implementation

- **Existing:** `tt/kda/config.py` defines the immutable program configurations and Kimi-K3 factory.
- **Existing:** `tt/kda/kda.py` owns SP/TP axes, CCL resources, and output reduce-scatter.
- **Existing:** `tt/kda/convolution.py` and `tt/kda/recurrence.py` launch synchronous SP-axis all-gathers without an explicit topology. TTNN deprecates topology on that interface, so the implementation must use the supported async collective when making topology explicit.
- **Existing:** `tests/kda/utils.py` constructs production-dimension KDA layers and inputs.
- **Existing:** `tests/kda/layer/test_acceptance.py` covers independent-reference accuracy and bitwise repeatability.
- **Existing:** `tests/kda/components/test_recurrence.py` asserts deterministic output and stable program-cache count.
- **Existing:** `tests/kda/perf/test_layer_perf.py` captures one trace and measures five synchronized groups of ten replays.
- **Existing:** component and distributed tests configure their multi-device fabric locally.

## Proposed changes

### KDA configuration

Modify `tt/kda/config.py`:

- remove `KDAProgramConfig.tp_ccl_topology`;
- remove the topology argument from `kimi_k3_program_config()`;
- keep compute fidelity, dtypes, chunking, and recurrence tuning unchanged.

This realizes the topology-ownership contract and intentionally breaks callers that still attempt to tune physical topology.

### KDA layer

Modify `tt/kda/kda.py`:

- add the required constructor parameter `topology: tuple[ttnn.Topology, ttnn.Topology]` with no default;
- validate that it is a tuple of exactly two `ttnn.Topology` values; reject scalar shorthand even when both axes would be Linear or Ring;
- after validating/storing axes, freeze `self.sp_ccl_topology = topology[sp_axis]` and `self.tp_ccl_topology = topology[tp_axis]`;
- in `_project_output`, use `self.tensor_parallel_axis` for all semaphore/link lookups and `cluster_axis` regardless of SP size;
- pass `self.tp_ccl_topology` to TP reduce-scatter;
- require a CCL manager whenever either mesh axis is distributed, and pass it with `self.sp_ccl_topology` into the convolution-carry and recurrence-summary helpers;
- retain the TP-size-one branch and current output memory/layout behavior.

Do not alter C++ collectives. If a retained failure appears below this boundary, stop and raise a design conflict with its standalone reproducer.

### SP collective helpers

Modify `tt/kda/convolution.py` and `tt/kda/recurrence.py`:

- add required scalar `topology: ttnn.Topology` and `tt_ccl: TT_CCL` arguments to the SP exchange/composition boundary;
- replace synchronous `ttnn.all_gather` with `ttnn.experimental.all_gather_async`, supplying the manager's all-gather semaphore, barrier semaphore, link count, the existing explicit `sequence_parallel_axis`, and the selected topology;
- do not accept a two-axis tuple at these leaf helpers: each launches CCLs on exactly one already-selected axis.

### Construction and contract tests

Modify all direct `ttKDA` construction sites, led by `tests/kda/utils.py` and `tests/kda/test_infrastructure.py`:

- call `kimi_k3_program_config()` without topology;
- derive the tuple at the composition boundary using `per_axis_topology(device_params["fabric_config"])` where the profile is available, or `per_axis_topology()` after that fixture has opened the mesh;
- pass the tuple explicitly to `ttKDA`; single-device cases pass the shared resolver's disabled-fabric result `(Linear, Linear)` without opening fabric;
- replace the program-config topology override test with explicit assertions for the shared mapping's plain, TorusX, TorusY, and TorusXY results;
- add focused KDA construction tests proving axis 0 and axis 1 select the corresponding tuple elements and proving scalar/wrong-length inputs fail;
- exercise swapped SP/TP assignments with an asymmetric tuple to prove tuple order remains physical rather than `(SP, TP)`.

### Fabric profile migration

Change every multi-device profile under `tests/kda/` from inline Fabric1D or `fabric_1d_device_params()` to `fabric2d_device_params()`. This includes convolution, weights, recurrence, distributed layer, acceptance, and performance tests. Preserve `torus_xy_device_params()` cases and feasibility markers.

Add a source-level collection assertion or small inventory test only if it is stable enough to prevent Fabric1D from silently returning; otherwise rely on a repository `rg` check in review/CI instructions.

### Durable regression coverage

Fold useful probe behavior into existing tests, then delete `tests/kda/layer/test_fabric2d_rca.py`:

- add SP1xTP8 to the synthetic Fabric2D accuracy/determinism matrix if runtime permits; otherwise keep the real-weight SP1 acceptance case and add a smaller deterministic synthetic regression;
- retain SP1xTP8, SP2xTP4, and SP4xTP2 in the trace/performance matrix;
- assert both selected topologies and axes in a focused contract test rather than parsing logs;
- retain TorusXY SP8xTP4 as hardware-gated acceptance.

### Performance baselines

Recalibrate `tests/kda/perf/test_layer_perf.py` on the designated LoudBox using the existing five-session policy. Record revision, fabric profile, host class, medians, and raw per-session medians. Do not promote the prototype's single-session values. Keep Galaxy TorusXY calibration separate.

## Control and data flow

1. Test/application chooses fabric, derives `(dim0 topology, dim1 topology)` with `per_axis_topology()`, and opens the matching mesh.
2. Caller constructs `ttKDA` with that tuple, SP/TP axes, and a CCL manager.
3. Constructor validates the tuple and freezes its SP- and TP-selected scalar values.
4. Forward computes local KDA output; SP carry exchanges launch async all-gather with the selected SP topology, explicit SP axis, and resources from the existing CCL manager.
5. If TP size > 1, output projection obtains semaphore handles and link count for the TP axis and launches reduce-scatter with the selected TP topology and same explicit axis.
6. The result remains sequence-sharded over SP and hidden-width-sharded over TP.

Failure paths:

- invalid/equal axes fail at construction as today;
- scalar, malformed, or non-`ttnn.Topology` tuple values fail at construction;
- missing CCL with TP > 1 or SP > 1 fails at construction;
- unknown torus/ring fabric mapping warns and chooses Linear at the caller's shared-helper boundary;
- a mismatch between the passed tuple and opened fabric cannot be detected cheaply by KDA; integration tests must derive both from the same profile value.

## Implementation sequence

1. Add focused required-tuple, tuple-order, topology-selection, and explicit-axis contract tests, initially failing on baseline behavior.
2. Remove the program-config topology API and update direct callers.
3. Thread the selected SP topology through convolution and recurrence; thread the selected TP topology through output projection.
4. Derive the tuple at every composition root and make all CCL axes explicit.
5. Migrate component/distributed/acceptance profiles to Fabric2D; validate in small groups through the safe wrapper.
6. Migrate trace/performance profiles and run the three-layout liveness/correctness suite.
7. Run TorusXY on a qualifying 8x4 host.
8. Calibrate Fabric2D performance over five independent sessions and commit baselines separately.
9. Remove the throwaway probe, run formatting/static checks, then run the final hardware matrix.

Optional follow-up: evaluate moving KDA from the transformer CCL manager to the DeepSeek CCL manager after API/resource equivalence is independently demonstrated.

## Validation

All hardware pytest commands must use `scripts/run_safe_pytest.sh`.

| Contract | Validation |
| --- | --- |
| Required topology API | scalar, omitted, short, and long inputs fail; valid two-element tuple succeeds |
| Per-axis topology | unit matrix for plain, TorusX, TorusY, TorusXY plus swapped-axis KDA selection |
| Explicit CCL axes | op-capture assertions for SP all-gathers and TP reduce-scatter in both logical-axis orientations |
| Numerical correctness | synthetic T=5120 and real-weight acceptance; output/recurrent/convolution PCC ≥ 0.9995 |
| Layout coverage | SP1xTP8, SP2xTP4, SP4xTP2 eager and trace |
| Determinism/cache | three bit-identical runs; cache count unchanged after warm-up |
| Torus | SP8xTP4 TorusXY on qualifying 8x4 hardware; skips reported as missing coverage |
| Performance | five independent sessions per Fabric2D LoudBox layout, existing 3% gate after calibration |
| Inventory | `rg -n 'FABRIC_1D|fabric_1d' models/demos/deepseek_v3_d_p/tests/kda` returns no matches |

Record exact command, commit, host, firmware/KMD, pass/fail/skip counts, PCC values, trace samples, JIT/cache evidence, and teardown/recovery result.

## Risks and unknowns

- **Unknown:** no local TorusXY execution was possible; do not close the permanent implementation work without it unless the owner explicitly waives that criterion.
- **Risk:** plain Fabric2D Linear increases SP1 trace latency relative to the currently working Ring request; calibration and any optimization must be separate commits.
- **Risk:** importing only the shared topology helper while retaining the older manager leaves two CCL implementations in the dependency graph. It is deliberate scope control, not the desired long-term consolidation.
- **Risk:** callers can pass a tuple inconsistent with the opened fabric. Keep fabric choice and tuple derivation adjacent at composition roots and cover them through profile-driven integration tests.
- **Assumed:** configured fabric is stable for the lifetime of constructed layers.
- **Required response:** if the historical timeout reappears, preserve complete logs and environment identity before changing C++ or recovery code.

## Implementation validation

The approved design was implemented in `da25439f8c3` and calibrated on `bh-lb-42-special-mvasilijevic-for-reservation-121898` with eight Blackhole p150b devices, firmware 19.5.0.0, KMD 2.4.1, and IOMMU enabled. Each calibration session was a separate safe-pytest process/device initialization. Values below are per-session medians in milliseconds; the reference is the median of those five medians.

| Layout | Session medians (ms) | Reference (ms) | Delta from prototype reference |
| --- | --- | ---: | ---: |
| SP1xTP8 | 11.067, 11.057, 11.051, 11.059, 11.048 | 11.057 | -0.04% |
| SP2xTP4 | 9.922, 9.921, 9.925, 9.920, 9.915 | 9.921 | +0.31% |
| SP4xTP2 | 10.259, 10.261, 10.257, 10.255, 10.251 | 10.257 | +2.01% |

The implementation is performance-neutral for SP1xTP8 and SP2xTP4 at this measurement resolution; SP4xTP2 is 2.01% slower than the single-session throwaway prototype and remains within the existing 3% gate. Against the earlier SP1xTP8 Ring request (9.672–9.703 ms), the supported plain-Fabric2D Linear path remains approximately 14% slower; this is a topology-policy cost, not an async-SP-gather regression, because SP1 performs no SP gather.
