# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Throwaway probe for the KDA SP1xTP8 Fabric 2D reduce-scatter failure."""

from __future__ import annotations

import os

import pytest

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.deepseek_v3_d_p.reference.kda import kda_forward_reference
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric2d_device_params
from models.demos.deepseek_v3_d_p.tests.kda.perf.test_layer_perf import _trace_wall_samples_ms
from models.demos.deepseek_v3_d_p.tests.kda.utils import (
    check_kimi_k3_accuracy,
    make_kimi_k3_device_case,
    make_synthetic_kimi_k3_test_case,
)
from tests.ttnn.unit_tests.operations.ccl.blackhole_CI.box.nightly.test_minimal_reduce_scatter_async_bh import (
    run_reduce_scatter_impl,
)

pytestmark = [run_for_blackhole(), pytest.mark.timeout(900)]


def _selected_topology() -> ttnn.Topology:
    value = os.environ.get("KDA_RCA_TOPOLOGY")
    if value == "linear":
        return ttnn.Topology.Linear
    if value == "ring":
        return ttnn.Topology.Ring
    raise ValueError("set KDA_RCA_TOPOLOGY to 'linear' or 'ring'")


def _selected_cluster_axis() -> int | None:
    value = os.environ.get("KDA_RCA_CLUSTER_AXIS")
    if value == "none":
        return None
    if value == "tp":
        return 1
    raise ValueError("set KDA_RCA_CLUSTER_AXIS to 'none' or 'tp'")


@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize("device_params", [fabric2d_device_params()], indirect=True)
def test_kda_reduce_scatter_topology_axis_matrix(mesh_device: ttnn.MeshDevice) -> None:
    """Run exactly one recovery-isolated matrix entry selected by environment."""
    topology = _selected_topology()
    cluster_axis = _selected_cluster_axis()
    print("KDA_FABRIC2D_RCA=" f"mesh={tuple(mesh_device.shape)},topology={topology},cluster_axis={cluster_axis}")
    run_reduce_scatter_impl(
        mesh_device,
        num_devices=8,
        rs_input_shape=[1, 1, 32, 256],
        dim=3,
        num_links=1,
        rs_input_dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mem_config_input=ttnn.DRAM_MEMORY_CONFIG,
        mem_config_rs=ttnn.DRAM_MEMORY_CONFIG,
        rs_topology=topology,
        enable_trace=False,
        cluster_axis=cluster_axis,
        use_barrier=True,
        use_persistent_buffers=False,
    )


@pytest.mark.parametrize(
    "mesh_device,tensor_parallel_axis",
    [((1, 8), 1), ((2, 4), 1), ((2, 4), 0)],
    indirect=["mesh_device"],
    ids=["SP1xTP8", "SP2xTP4", "SP4xTP2"],
)
@pytest.mark.parametrize("device_params", [fabric2d_device_params()], indirect=True)
def test_kda_fabric2d_full_layer(mesh_device: ttnn.MeshDevice, tensor_parallel_axis: int) -> None:
    """Probe all eight-device KDA layouts with production dimensions."""
    sequence = int(os.environ.get("KDA_RCA_SEQUENCE", "128"))
    case = make_synthetic_kimi_k3_test_case(sequence=sequence)
    golden_output, golden_state = kda_forward_reference(case.hidden, case.state_dict, case.config)
    layer, hidden = make_kimi_k3_device_case(
        mesh_device,
        case,
        tensor_parallel_axis=tensor_parallel_axis,
        cache_weights=False,
    )
    state = layer.allocate_state()
    output, state = layer.forward(hidden, state)
    ttnn.synchronize_device(mesh_device)
    pcc = check_kimi_k3_accuracy(
        f"KDA RCA FABRIC_2D mesh={tuple(mesh_device.shape)} tp_axis={tensor_parallel_axis} T={sequence}",
        case,
        golden_output,
        golden_state,
        state,
        output,
        mesh_device,
        tensor_parallel_axis=tensor_parallel_axis,
        pcc_threshold=0.9995,
    )
    print(
        "KDA_FABRIC2D_FULL_LAYER="
        f"sequence={sequence},mesh={tuple(mesh_device.shape)},topology={layer.tp_ccl_topology},"
        f"tp_axis={tensor_parallel_axis},pcc={pcc}"
    )


@pytest.mark.parametrize(
    "mesh_device,tensor_parallel_axis",
    [((1, 8), 1), ((2, 4), 1), ((2, 4), 0)],
    indirect=["mesh_device"],
    ids=["SP1xTP8", "SP2xTP4", "SP4xTP2"],
)
@pytest.mark.parametrize("device_params", [fabric2d_device_params()], indirect=True)
def test_kda_trace_replay(mesh_device: ttnn.MeshDevice, tensor_parallel_axis: int) -> None:
    """Exercise the same capture plus 5x10 replay loop as the failing perf case."""
    sequence = int(os.environ.get("KDA_RCA_SEQUENCE", "5120"))
    case = make_synthetic_kimi_k3_test_case(sequence=sequence)
    layer, hidden = make_kimi_k3_device_case(
        mesh_device,
        case,
        tensor_parallel_axis=tensor_parallel_axis,
        cache_weights=False,
    )
    samples_ms, _ = _trace_wall_samples_ms(mesh_device, layer, hidden, repetitions=10)
    print(
        "KDA_FABRIC2D_TRACE_REPLAY="
        f"sequence={sequence},mesh={tuple(mesh_device.shape)},topology={layer.tp_ccl_topology},"
        f"tp_axis={tensor_parallel_axis},samples_ms={samples_ms}"
    )
