#!/usr/bin/env python3
"""Ascend NPU trace -> operator replay program generator.

读取 execution trace + kineto trace，生成可在昇腾 NPU 环境执行的算子流重放脚本。
"""

from __future__ import annotations

import argparse
import json
import textwrap
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


DTYPE_MAP = {
    "long int": "torch.int64",
    "int": "torch.int32",
    "float": "torch.float32",
    "double": "torch.float64",
    "half": "torch.float16",
    "bool": "torch.bool",
    "bfloat16": "torch.bfloat16",
}


@dataclass
class ExecutionNode:
    node_id: int
    name: str
    ctrl_deps: int
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    attrs: Dict[str, Any]


@dataclass
class KinetoEvent:
    name: str
    ts: float
    dur: float
    cat: str
    args: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReplayOp:
    idx: int
    name: str
    node_id: int
    duration_us: float
    input_shapes: List[List[int]]
    input_types: List[str]
    output_shapes: List[List[int]]
    output_types: List[str]
    input_values: List[Any]
    output_values: List[Any]


class TraceLoader:
    @staticmethod
    def load_execution(path: Path) -> List[ExecutionNode]:
        data = json.loads(path.read_text(encoding="utf-8"))
        nodes = []
        for n in data.get("nodes", []):
            attrs = {a.get("name"): a.get("value") for a in n.get("attrs", [])}
            nodes.append(
                ExecutionNode(
                    node_id=n.get("id"),
                    name=n.get("name", ""),
                    ctrl_deps=n.get("ctrl_deps", -1),
                    inputs=n.get("inputs", {}),
                    outputs=n.get("outputs", {}),
                    attrs=attrs,
                )
            )
        return nodes

    @staticmethod
    def load_kineto(path: Path) -> List[KinetoEvent]:
        raw = json.loads(path.read_text(encoding="utf-8"))
        events: List[KinetoEvent] = []
        for e in raw:
            if e.get("ph") != "X":
                continue
            events.append(
                KinetoEvent(
                    name=e.get("name", ""),
                    ts=float(e.get("ts", 0.0)),
                    dur=float(e.get("dur", 0.0)),
                    cat=e.get("cat", ""),
                    args=e.get("args", {}),
                )
            )
        events.sort(key=lambda x: x.ts)
        return events


class TraceAligner:
    @staticmethod
    def align(nodes: List[ExecutionNode], events: List[KinetoEvent]) -> List[ReplayOp]:
        by_name: Dict[str, List[KinetoEvent]] = defaultdict(list)
        for e in events:
            by_name[e.name].append(e)

        replay_ops: List[ReplayOp] = []
        for n in nodes:
            if not n.name or n.name.startswith("["):
                continue
            if n.name.startswith("## process_group:init"):
                continue

            matched = by_name.get(n.name, [])
            duration = 0.0
            if matched:
                duration = matched.pop(0).dur

            replay_ops.append(
                ReplayOp(
                    idx=len(replay_ops),
                    name=n.name,
                    node_id=n.node_id,
                    duration_us=duration,
                    input_shapes=n.inputs.get("shapes", []),
                    input_types=n.inputs.get("types", []),
                    output_shapes=n.outputs.get("shapes", []),
                    output_types=n.outputs.get("types", []),
                    input_values=n.inputs.get("values", []),
                    output_values=n.outputs.get("values", []),
                )
            )
        return replay_ops


def _normalize_shape(shape: Any) -> List[int]:
    if not isinstance(shape, list):
        return [1]
    if len(shape) == 0:
        return [1]
    # 过滤 SymInt list 这类元信息（例如 [[], []]）
    if all(isinstance(x, list) for x in shape):
        return [1]
    out = []
    for d in shape:
        if isinstance(d, int):
            out.append(max(1, d))
    return out or [1]


def _dtype_expr(type_desc: str) -> str:
    if type_desc.startswith("Tensor("):
        inner = type_desc.removeprefix("Tensor(").removesuffix(")")
        return DTYPE_MAP.get(inner.strip(), "torch.float32")
    return "torch.float32"


class ReplayProgramBuilder:
    def __init__(self, replay_ops: List[ReplayOp]) -> None:
        self.replay_ops = replay_ops

    def build(self) -> str:
        op_dicts = []
        for op in self.replay_ops:
            op_dicts.append(
                {
                    "idx": op.idx,
                    "name": op.name,
                    "node_id": op.node_id,
                    "duration_us": op.duration_us,
                    "input_shapes": [_normalize_shape(s) for s in op.input_shapes],
                    "input_types": op.input_types,
                    "output_shapes": [_normalize_shape(s) for s in op.output_shapes],
                    "output_types": op.output_types,
                    "input_values": op.input_values,
                    "output_values": op.output_values,
                }
            )

        payload = json.dumps(op_dicts, ensure_ascii=False, indent=2)

        return textwrap.dedent(
            f"""#!/usr/bin/env python3
# Auto-generated by trace2replay.py
import json
import time
import torch

try:
    import torch_npu  # noqa: F401
except Exception as exc:
    raise RuntimeError("需要在昇腾 NPU 环境安装 torch-npu 才能运行重放程序") from exc

OPS = json.loads(r'''{payload}''')

def _dtype_expr(type_desc: str):
    m = {{
        "long int": torch.int64,
        "int": torch.int32,
        "float": torch.float32,
        "double": torch.float64,
        "half": torch.float16,
        "bool": torch.bool,
        "bfloat16": torch.bfloat16,
    }}
    if type_desc.startswith("Tensor("):
        key = type_desc[len("Tensor("):-1]
        return m.get(key.strip(), torch.float32)
    return torch.float32

def _make_tensor(shape, type_desc, device):
    dtype = _dtype_expr(type_desc)
    if dtype is torch.bool:
        t = torch.randint(0, 2, shape, dtype=torch.int32, device=device)
        return t.bool()
    return torch.randn(*shape, dtype=dtype, device=device)

def replay(device="npu:0", dryrun=False):
    tensor_map = {{}}
    started = time.perf_counter()
    for op in OPS:
        name = op["name"]
        ivals = op.get("input_values", [])
        ovals = op.get("output_values", [])
        ishapes = op.get("input_shapes", [])
        itypes = op.get("input_types", [])
        dur_us = float(op.get("duration_us", 0.0))

        def fetch_input(i):
            if i < len(ivals) and isinstance(ivals[i], list) and ivals[i] and isinstance(ivals[i][0], int):
                tid = ivals[i][0]
                if tid in tensor_map:
                    return tensor_map[tid]
            shape = ishapes[i] if i < len(ishapes) else [1]
            tdesc = itypes[i] if i < len(itypes) else "Tensor(float)"
            return _make_tensor(shape, tdesc, device)

        out = None
        if name == "aten::view":
            x = fetch_input(0)
            target = ivals[1] if len(ivals) > 1 and isinstance(ivals[1], list) else [-1]
            target = [int(v) for v in target]
            out = x.view(*target)
        elif name == "aten::to":
            x = fetch_input(0)
            out = x.to(device)
        elif name == "aten::flatten":
            x = fetch_input(0)
            start_dim = int(ivals[1]) if len(ivals) > 1 and isinstance(ivals[1], int) else 0
            end_dim = int(ivals[2]) if len(ivals) > 2 and isinstance(ivals[2], int) else -1
            out = torch.flatten(x, start_dim=start_dim, end_dim=end_dim)
        elif name in ("aten::lt",):
            x = fetch_input(0)
            scalar = ivals[1] if len(ivals) > 1 and isinstance(ivals[1], (int, float)) else 0
            out = torch.lt(x, scalar)
        elif name in ("aten::ge",):
            x = fetch_input(0)
            scalar = ivals[1] if len(ivals) > 1 and isinstance(ivals[1], (int, float)) else 0
            out = torch.ge(x, scalar)
        elif name in ("aten::bitwise_or", "aten::__or__"):
            x = fetch_input(0)
            y = fetch_input(1)
            out = torch.bitwise_or(x, y)
        elif name == "aten::clone":
            out = fetch_input(0).clone()
        elif name == "aten::copy_":
            dst = fetch_input(0)
            src = fetch_input(1)
            out = dst.copy_(src)
        else:
            # 未实现算子用小型矩阵乘模拟负载
            a = torch.randn(256, 256, device=device)
            b = torch.randn(256, 256, device=device)
            out = a @ b

        if ovals and isinstance(ovals[0], list) and ovals[0] and isinstance(ovals[0][0], int):
            tensor_map[ovals[0][0]] = out

        if not dryrun and dur_us > 0:
            target = time.perf_counter() + dur_us / 1e6
            while time.perf_counter() < target:
                _ = out

    torch.npu.synchronize()
    elapsed = (time.perf_counter() - started) * 1000
    print(f"Replay complete. ops={{len(OPS)}}, elapsed={{elapsed:.3f}} ms")

if __name__ == "__main__":
    replay()
            """
        ).strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="execution trace + kineto trace -> Ascend NPU replay program")
    parser.add_argument("--execution-trace", required=True, type=Path, help="execution trace JSON path")
    parser.add_argument("--kineto-trace", required=True, type=Path, help="kineto trace JSON path")
    parser.add_argument("--output", required=True, type=Path, help="generated replay python file")
    args = parser.parse_args()

    nodes = TraceLoader.load_execution(args.execution_trace)
    events = TraceLoader.load_kineto(args.kineto_trace)
    replay_ops = TraceAligner.align(nodes, events)

    content = ReplayProgramBuilder(replay_ops).build()
    args.output.write_text(content, encoding="utf-8")

    print(f"Loaded execution nodes: {len(nodes)}")
    print(f"Loaded kineto events: {len(events)}")
    print(f"Generated replay ops: {len(replay_ops)}")
    print(f"Replay program written to: {args.output}")


if __name__ == "__main__":
    main()
