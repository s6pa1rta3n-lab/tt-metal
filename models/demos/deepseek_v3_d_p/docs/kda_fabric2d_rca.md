# KDA Fabric2D SP1xTP8 RCA

Work: `tt-metal_tracker-66t.2`
Baseline: PR7 local head `68f40f4228844574cbe5fadd14ee79e4af610ec4`
Hardware: `bh-lb-42-special-mvasilijevic-for-reservation-121898`, 8 Blackhole p150b devices, firmware 19.5.0.0, KMD 2.4.1, IOMMU enabled

## Verdict

**Observed:** the reported SP1xTP8 Fabric2D hang is not reproducible on this PR7 checkout and current hardware/software stack. The unmodified KDA path completes both eager production-shape execution and the exact performance-test trace pattern (capture, first replay, then 5 samples of 10 replays) with clean device teardown.

**Observed:** the proposed root cause in the parent bead is falsified on this stack. All four `topology × cluster_axis` reduce-scatter cases pass. In particular, `Ring + None`, the current KDA combination, neither hangs nor produces incorrect output.

**Unknown:** the historical failure's root cause cannot be recovered because the repository retains only a comment, not the failing revision, full log, host identity, firmware/runtime versions, or reproducer output. It must not be attributed to a topology mismatch as an established fact.

## Current-system evidence

- **Observed at baseline:** KDA selected Ring when SP size was one in `tests/kda/utils.py` and passed `cluster_axis=None` from the output projection in `tt/kda/kda.py`. Git blame attributes both choices to the original KDA layer commit; neither was a recent direct fix.
- **Observed:** MLA treats collective topology as per-axis state and always passes its TP axis to output reduce-scatter (`tt/mla/mla.py:452-466`, `tt/mla/mla.py:1263-1273`).
- **Observed:** the shared DeepSeek mapping defines plain Fabric2D as `(Linear, Linear)`, TorusX as `(Linear, Ring)`, TorusY as `(Ring, Linear)`, and TorusXY as `(Ring, Ring)` (`tt/tt_ccl.py:372-413`).
- **Observed:** TTNN normalizes requested Ring/Linear topology before dispatch, but explicit-axis Ring boundary selection is based on mesh extent, while physical wrap detection is a separate helper (`ttnn/cpp/ttnn/operations/ccl/ccl_common.cpp:84-115`, `ttnn/cpp/ttnn/operations/ccl/ccl_common.cpp:180-207`). This makes source-only reasoning about a guaranteed missing-wrap deadlock insufficient.

## Reproduction matrix

The probe is `tests/kda/layer/test_fabric2d_rca.py`. Every device run used `scripts/run_safe_pytest.sh`, which serialized access, imposed a five-second dispatch timeout, and performed recovery when marked dirty.

| Fabric2D case | Result | PCC | Pytest call | Recovery |
| --- | --- | --- | ---: | --- |
| TP axis, Linear | collective passed | 0.9998594876 | 3.06 s | clean on non-dev repeat |
| no axis, Linear | collective passed | 0.9998594876 | 0.05 s | clean |
| TP axis, Ring | collective passed | 0.9998958423 | 2.53 s | clean |
| no axis, Ring | collective passed | 0.9998958423 | 0.10 s | clean |

Commands differed only in the two environment selectors:

```bash
KDA_RCA_TOPOLOGY={linear|ring} KDA_RCA_CLUSTER_AXIS={tp|none} \
  scripts/run_safe_pytest.sh \
  models/demos/deepseek_v3_d_p/tests/kda/layer/test_fabric2d_rca.py \
  -k topology_axis_matrix -s -q
```

The first `TP axis + Linear` run under `--dev` passed the operation and PCC check, then aborted during teardown while restoring Ethernet core device 0 virtual core 26-25 (`SAFE_PYTEST_RESULT: FAIL`, exit 134). A safe-wrapper reset followed, and the identical non-`--dev` run passed cleanly. **Inferred:** this one-off teardown fault is evidence that instrumentation/recovery state can produce the historical failure signature after successful computation; it is not sufficient evidence that this was the old failure.

## Full-layer evidence

Before the prototype:

- T=128 eager SP1xTP8: output 0.9999516582, recurrent 0.9997899785, convolution 0.9999966537; 34.55 s call; pass and clean teardown.
- T=5120 eager SP1xTP8: output 0.9999510499, recurrent 0.9997580821, convolution 0.9999966916; 132.81 s call; pass and clean teardown.
- T=5120 SP1xTP8 trace: 50 timed replays at 9.672–9.703 ms; 8.43 s call; pass and clean teardown.

Commands:

```bash
KDA_RCA_SEQUENCE=128 scripts/run_safe_pytest.sh \
  models/demos/deepseek_v3_d_p/tests/kda/layer/test_fabric2d_rca.py -k full_layer -s -q
KDA_RCA_SEQUENCE=5120 scripts/run_safe_pytest.sh \
  models/demos/deepseek_v3_d_p/tests/kda/layer/test_fabric2d_rca.py -k full_layer -s -q
KDA_RCA_SEQUENCE=5120 scripts/run_safe_pytest.sh \
  models/demos/deepseek_v3_d_p/tests/kda/layer/test_fabric2d_rca.py -k trace_replay -s -q
```

## Causal conclusion and blast radius

No current product failure was isolated, so no causal fix to Python or C++ is justified. The defensible defect is a contract inconsistency: KDA derives topology from SP size and sometimes omits the TP axis, while MLA derives topology from the active fabric and names the collective axis. This inconsistency is independently worth fixing because it makes topology intent implicit and diverges from the model's established CCL contract.

The prototype therefore stays in DeepSeek KDA Python and KDA tests. It reuses the existing shared DeepSeek topology mapping; no C++ modification is supported by the evidence. Shared CCL behavior should only be changed if a separately retained reproducer fails below the KDA boundary.

## Remaining unknowns

- No 8x4 TorusXY host was available; its existing test was skipped as infeasible on this 8-device host.
- The historical failing environment and exact failure phase are unknown.
- The one-off `--dev` teardown recovery failure was not pursued because it did not reproduce without instrumentation and is outside the KDA data path.
