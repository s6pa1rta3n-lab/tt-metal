# KDA Fabric2D Support Development Spec

Work: `tt-metal_tracker-66t.5`
Approved direction: [kda_fabric2d_design.md](kda_fabric2d_design.md)
Evidence: [kda_fabric2d_rca.md](kda_fabric2d_rca.md)

## Goal

Implement the reviewed KDA contract in which active fabric determines per-axis topology, TP reduce-scatter always names its axis, and all multi-device KDA tests default to Fabric2D. Do not carry the diagnostic probe or one-session performance numbers into the production patch.

Out of scope: C++/firmware changes, CCL-manager consolidation, Fabric1D comparison coverage, and investigation of the unretained historical failure.

## Design constraints

- **Required:** `per_axis_topology()` is the only fabric-to-topology mapping used by KDA.
- **Required:** topology is resolved once at construction and cannot be supplied through program tuning.
- **Required:** semaphore selection, link selection, and reduce-scatter use the same TP axis.
- **Required:** plain Fabric2D is the default for multi-device KDA tests; torus is explicit; single-device tests need no fabric.
- **Required:** numerical, state, determinism, cache, trace, and fabric-specific performance contracts remain gated.

## Existing implementation

- **Existing:** `tt/kda/config.py` defines the immutable program configurations and Kimi-K3 factory.
- **Existing:** `tt/kda/kda.py` owns SP/TP axes, CCL resources, and output reduce-scatter.
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

- import the shared DeepSeek `per_axis_topology` helper;
- after validating/storing axes, assign the immutable TP topology from the active fabric and `tp_axis`;
- in `_project_output`, use `self.tensor_parallel_axis` for all semaphore/link lookups and `cluster_axis` regardless of SP size;
- retain the TP-size-one branch and current output memory/layout behavior.

Do not alter C++ collectives. If a retained failure appears below this boundary, stop and raise a design conflict with its standalone reproducer.

### Construction and contract tests

Modify `tests/kda/utils.py` and `tests/kda/test_infrastructure.py`:

- call `kimi_k3_program_config()` without topology;
- replace the program-config topology override test with explicit assertions for the shared mapping's plain, TorusX, TorusY, and TorusXY results;
- add a focused KDA construction test proving axis 0 and axis 1 select the corresponding tuple element. Mock only the topology resolver if fabric cannot be changed safely in a unit fixture.

### Fabric profile migration

Change every multi-device profile under `tests/kda/` from inline Fabric1D or `fabric_1d_device_params()` to `fabric2d_device_params()`. This includes convolution, weights, recurrence, distributed layer, acceptance, and performance tests. Preserve `torus_xy_device_params()` cases and feasibility markers.

Add a source-level collection assertion or small inventory test only if it is stable enough to prevent Fabric1D from silently returning; otherwise rely on a repository `rg` check in review/CI instructions.

### Durable regression coverage

Fold useful probe behavior into existing tests, then delete `tests/kda/layer/test_fabric2d_rca.py`:

- add SP1xTP8 to the synthetic Fabric2D accuracy/determinism matrix if runtime permits; otherwise keep the real-weight SP1 acceptance case and add a smaller deterministic synthetic regression;
- retain SP1xTP8, SP2xTP4, and SP4xTP2 in the trace/performance matrix;
- assert the resolved topology and TP axis in a focused contract test rather than parsing logs;
- retain TorusXY SP8xTP4 as hardware-gated acceptance.

### Performance baselines

Recalibrate `tests/kda/perf/test_layer_perf.py` on the designated LoudBox using the existing five-session policy. Record revision, fabric profile, host class, medians, and raw per-session medians. Do not promote the prototype's single-session values. Keep Galaxy TorusXY calibration separate.

## Control and data flow

1. Test/application configures fabric and opens the mesh.
2. Caller constructs `ttKDA` with SP/TP axes and a CCL manager.
3. Constructor queries shared active-fabric mapping and freezes topology for the selected TP axis.
4. Forward computes local KDA output.
5. If TP size > 1, output projection obtains semaphore handles and link count for the TP axis and launches reduce-scatter with the same axis/topology.
6. The result remains sequence-sharded over SP and hidden-width-sharded over TP.

Failure paths:

- invalid/equal axes fail at construction as today;
- missing CCL with TP > 1 fails at construction as today;
- unknown torus/ring fabric mapping warns and chooses Linear through the shared helper;
- a fabric change after construction is unsupported; document or assert this only if the runtime exposes a cheap stable check.

## Implementation sequence

1. Add focused topology/axis contract tests, initially failing on baseline behavior.
2. Remove the program-config topology API and update direct callers.
3. Resolve topology from active fabric and make TP axis explicit.
4. Migrate component/distributed/acceptance profiles to Fabric2D; validate in small groups through the safe wrapper.
5. Migrate trace/performance profiles and run the three-layout liveness/correctness suite.
6. Run TorusXY on a qualifying 8x4 host.
7. Calibrate Fabric2D performance over five independent sessions and commit baselines separately.
8. Remove the throwaway probe, run formatting/static checks, then run the final hardware matrix.

Optional follow-up: evaluate moving KDA from the transformer CCL manager to the DeepSeek CCL manager after API/resource equivalence is independently demonstrated.

## Validation

All hardware pytest commands must use `scripts/run_safe_pytest.sh`.

| Contract | Validation |
| --- | --- |
| Per-axis topology | unit matrix for plain, TorusX, TorusY, TorusXY plus KDA axis selection |
| Explicit TP axis | focused KDA construction/op capture assertion; both TP-axis orientations |
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
- **Assumed:** active fabric is stable for the lifetime of constructed layers.
- **Required response:** if the historical timeout reappears, preserve complete logs and environment identity before changing C++ or recovery code.
