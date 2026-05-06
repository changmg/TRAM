#!/usr/bin/env python3
"""
Lightweight Python approximate-multiplier simulator.

A NumPy bit-parallel replacement for the C++/ABC tool that lives at
~/Work/WIP/post-training-approximation/simulator/. Reads generic BLIF
(Berkeley spec — see simulator/reference/blif_format.pdf), exhaustively
simulates an N x N approximate multiplier, and prints error metrics plus
the lookup table in a format that matches the .txt reference files in
app_mults/evo_selected/.

BLIF parsing is delegated to a vendored copy of mario33881/blifparser
(MIT-licensed -- see simulator/blifparser/LICENSE.txt and
simulator/NOTICE.md for credit). This script adds the multi-model and
recursive .subckt flattening that blifparser does not provide on its own.
"""

import argparse
import os
import sys
import tempfile
from collections import defaultdict, deque

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from blifparser.blifparser import BlifParser  # noqa: E402


CONST_NAMES = {"$true": 1, "$false": 0, "$undef": 0}


# -----------------------------------------------------------------------------
# 1. BLIF ingestion: pre-split + per-model parse via blifparser
# -----------------------------------------------------------------------------

class FlatNode:
    __slots__ = ("out", "fanins", "cubes", "complement")

    def __init__(self, out, fanins, cubes, complement):
        self.out = out
        self.fanins = fanins
        self.cubes = cubes
        self.complement = complement


class ParsedModel:
    __slots__ = ("name", "inputs", "outputs", "names", "subckts")

    def __init__(self, name, inputs, outputs, names, subckts):
        self.name = name
        self.inputs = inputs
        self.outputs = outputs
        self.names = names      # list[FlatNode]
        self.subckts = subckts  # list[(modelname, dict[formal -> actual])]


def split_models(path):
    """Return [(model_name, model_text), ...]. Each text is a complete
    self-contained BLIF body for one .model block, terminated by .end so it
    can be fed to BlifParser as a standalone file."""
    with open(path, "r") as f:
        lines = f.readlines()

    blocks = []
    cur = []
    cur_name = None
    for ln in lines:
        stripped = ln.lstrip()
        if stripped.startswith(".model"):
            if cur_name is not None:
                if not any(l.lstrip().startswith(".end") for l in cur):
                    cur.append(".end\n")
                blocks.append((cur_name, "".join(cur)))
            cur = [ln]
            parts = stripped.split()
            cur_name = parts[1] if len(parts) > 1 else "<anonymous>"
        else:
            if cur_name is not None:
                cur.append(ln)
            # lines before the first .model are header comments — skip

    if cur_name is not None:
        if not any(l.lstrip().startswith(".end") for l in cur):
            cur.append(".end\n")
        blocks.append((cur_name, "".join(cur)))

    return blocks


def parse_one_model(text):
    """Run blifparser on a single-model BLIF string. Returns ParsedModel."""
    with tempfile.NamedTemporaryFile("w", suffix=".blif", delete=False) as tf:
        tf.write(text)
        tmp_path = tf.name
    try:
        blif = BlifParser(tmp_path).blif
    finally:
        os.unlink(tmp_path)

    fatal = [p for p in blif.problems if not p.startswith("WARNING")]
    if fatal:
        raise RuntimeError("blifparser errors:\n  " + "\n  ".join(fatal))
    if blif.latches:
        raise RuntimeError("BLIF contains .latch; sequential designs are not supported")

    name = blif.model.name if blif.model else "<anonymous>"
    inputs = list(blif.inputs.inputs) if blif.inputs else []
    outputs = list(blif.outputs.outputs) if blif.outputs else []

    names_nodes = []
    for nm in blif.booleanfunctions:
        if getattr(nm, "is_dontcare", False):
            continue
        cubes_in = []
        out_planes = set()
        for row in nm.truthtable:
            *in_plane, out_ch = row
            cubes_in.append("".join(in_plane))
            out_planes.add(out_ch)
        if not out_planes or out_planes == {"1"}:
            complement = False
        elif out_planes == {"0"}:
            complement = True
        else:
            raise RuntimeError(
                f".names {nm.output} has mixed output planes; not supported"
            )
        names_nodes.append(FlatNode(nm.output, list(nm.inputs), cubes_in, complement))

    subckts = []
    for sc in blif.subcircuits:
        mapping = {}
        for tok in sc.params:
            f, a = tok.split("=", 1)
            mapping[f] = a
        subckts.append((sc.modelname, mapping))

    return ParsedModel(name, inputs, outputs, names_nodes, subckts)


# -----------------------------------------------------------------------------
# 2. Recursive .subckt flattening
# -----------------------------------------------------------------------------

def flatten(top_name, registry):
    """Inline every .subckt instance recursively into `top_name`. Returns a
    flat list of FlatNode whose output names are unique. The tokens
    $true/$false/$undef are treated as model-global constants (one node each
    in the flat netlist)."""
    flat = []
    seen_outs = set()
    inst_counter = [0]

    def walk(model_name, name_remap):
        m = registry[model_name]
        for nd in m.names:
            new_out = name_remap.get(nd.out, nd.out)
            new_fanins = [name_remap.get(fi, fi) for fi in nd.fanins]
            if new_out in CONST_NAMES and new_out in seen_outs:
                continue
            if new_out in seen_outs:
                raise RuntimeError(
                    f"flatten: net '{new_out}' is driven more than once"
                )
            flat.append(FlatNode(new_out, new_fanins, nd.cubes, nd.complement))
            seen_outs.add(new_out)

        for submodel, formal_to_actual in m.subckts:
            if submodel not in registry:
                raise RuntimeError(
                    f"flatten: .subckt references unknown model '{submodel}'"
                )
            inst_counter[0] += 1
            prefix = f"$inst{inst_counter[0]}$"
            sub = registry[submodel]
            child_remap = {}
            for f, a in formal_to_actual.items():
                child_remap[f] = name_remap.get(a, a)
            all_wires = set()
            for nd in sub.names:
                all_wires.add(nd.out)
                all_wires.update(nd.fanins)
            for sm_sub, sm_map in sub.subckts:
                for f, a in sm_map.items():
                    all_wires.add(a)
            for wire in all_wires:
                if wire in child_remap or wire in CONST_NAMES:
                    continue
                child_remap[wire] = prefix + wire
            walk(submodel, child_remap)

    walk(top_name, {})
    return flat


def inject_const0_drivers(flat_nodes, pi_names, top_name):
    """Find every fanin in `flat_nodes` that has no driver (not a PI, not a
    declared constant token, and not the output of any node). For each such
    net, append a const-0 FlatNode to `flat_nodes` so the simulator has a
    well-defined value, and print the warning ABC emits in the same format
    used by the reference .txt files (first four names, then ' ...').
    Order is BLIF appearance order: a fanin's first occurrence as we walk
    `flat_nodes` in their original order."""
    driven = set(pi_names)
    driven.update(CONST_NAMES)
    driven.update(nd.out for nd in flat_nodes)
    seen = set()
    undriven = []
    for nd in flat_nodes:
        for fi in nd.fanins:
            if fi in driven or fi in seen:
                continue
            seen.add(fi)
            undriven.append(fi)
    if not undriven:
        return
    n = len(undriven)
    head = ", ".join(undriven[:4]) + (" ..." if n > 4 else "")
    print(
        f'Warning: Constant-0 drivers added to {n} non-driven nets in '
        f'network "{top_name}":'
    )
    print(head)
    for name in undriven:
        flat_nodes.append(FlatNode(name, [], [], False))


# -----------------------------------------------------------------------------
# 3. Bit-parallel simulation
# -----------------------------------------------------------------------------

# For PI i with i < 6, every uint64 word holds the same fixed mask. The mask
# encodes bit i of each frame index in the 64 frame slots packed into the
# word (frame index = word_index*64 + bit_position).
_LOW_PI_MASKS = np.array([
    0xAAAAAAAAAAAAAAAA,  # i = 0: bit 0 of (0..63) -> alternating 0,1
    0xCCCCCCCCCCCCCCCC,  # i = 1
    0xF0F0F0F0F0F0F0F0,  # i = 2
    0xFF00FF00FF00FF00,  # i = 3
    0xFFFF0000FFFF0000,  # i = 4
    0xFFFFFFFF00000000,  # i = 5
], dtype=np.uint64)

_FULL = np.uint64(0xFFFFFFFFFFFFFFFF)
_ZERO = np.uint64(0)


def make_pi_arrays(n_pi, n_words):
    """Build the uint64 array for each PI such that bit position
    (word*64 + b) of the array equals (frame >> i) & 1 where frame = word*64 + b."""
    arrs = []
    word_idx = np.arange(n_words, dtype=np.uint64)
    for i in range(n_pi):
        if i < 6:
            arr = np.full(n_words, _LOW_PI_MASKS[i], dtype=np.uint64)
        else:
            shift = np.uint64(i - 6)
            bit_per_word = (word_idx >> shift) & np.uint64(1)
            arr = bit_per_word * _FULL
        arrs.append(arr)
    return arrs


def eval_node(nd, fanin_arrs, n_words):
    out = np.zeros(n_words, dtype=np.uint64)
    for cube in nd.cubes:
        prod = None
        for ch, fi_arr in zip(cube, fanin_arrs):
            if ch == "-":
                continue
            v = (~fi_arr) if ch == "0" else fi_arr
            prod = v.copy() if prod is None else (prod & v)
        if prod is None:
            prod = np.full(n_words, _FULL, dtype=np.uint64)
        out |= prod
    if nd.complement:
        out = ~out
    return out


def simulate(flat_nodes, pi_names, po_names, n_pi):
    n_frame = 1 << n_pi
    n_words = max(1, n_frame // 64)

    sigs = {}
    pi_arrs = make_pi_arrays(n_pi, n_words)
    for i, name in enumerate(pi_names):
        sigs[name] = pi_arrs[i]

    by_out = {nd.out: nd for nd in flat_nodes}

    indeg = {nd.out: 0 for nd in flat_nodes}
    revdep = defaultdict(list)
    for nd in flat_nodes:
        deps = 0
        for fi in nd.fanins:
            if fi in sigs or fi in CONST_NAMES:
                continue
            if fi in by_out:
                revdep[fi].append(nd.out)
                deps += 1
            else:
                raise RuntimeError(
                    f"unresolved fanin '{fi}' in .names {nd.out}"
                )
        indeg[nd.out] = deps

    def ensure_const(name):
        if name in sigs:
            return
        if name in CONST_NAMES:
            sigs[name] = np.full(
                n_words, _FULL if CONST_NAMES[name] else _ZERO, dtype=np.uint64
            )

    queue = deque(nd.out for nd in flat_nodes if indeg[nd.out] == 0)
    while queue:
        name = queue.popleft()
        nd = by_out[name]
        fanin_arrs = []
        for fi in nd.fanins:
            if fi in CONST_NAMES:
                ensure_const(fi)
            if fi not in sigs:
                raise RuntimeError(
                    f"signal '{fi}' not yet computed for node '{name}'"
                )
            fanin_arrs.append(sigs[fi])
        sigs[name] = eval_node(nd, fanin_arrs, n_words)
        for downstream in revdep.get(name, ()):
            indeg[downstream] -= 1
            if indeg[downstream] == 0:
                queue.append(downstream)

    uncomputed = [k for k, v in indeg.items() if v != 0]
    if uncomputed:
        raise RuntimeError(
            f"cycle in netlist; uncomputed nodes (sample): {uncomputed[:5]}"
        )

    po_arrs = []
    for po in po_names:
        if po not in sigs:
            ensure_const(po)
        if po not in sigs:
            raise RuntimeError(f"PO '{po}' was not driven by any .names")
        po_arrs.append(sigs[po])

    return po_arrs, n_frame


# -----------------------------------------------------------------------------
# 4. Decode + metrics + LUT dump
# -----------------------------------------------------------------------------

def decode_outputs(po_arrs, n_frame):
    """Convert a list of PO uint64 arrays into a length-n_frame int64 array
    holding the unsigned integer output per frame."""
    outp = np.zeros(n_frame, dtype=np.int64)
    for k, arr in enumerate(po_arrs):
        # Force little-endian byte order so that bit 0 of each uint64 lands in
        # byte 0 of its view, matching frame index 0 within that 64-frame block.
        bits = np.unpackbits(
            arr.astype("<u8").view(np.uint8), bitorder="little"
        )[:n_frame]
        outp += bits.astype(np.int64) << np.int64(k)
    return outp


def signed_extend(x, width):
    half = 1 << (width - 1)
    full = 1 << width
    return np.where(x >= half, x - full, x)


def fmt_g(x):
    return f"{x:g}"


def main():
    # Treat a closed stdout (e.g. when piped through `head`) as a normal exit.
    try:
        from signal import SIG_DFL, SIGPIPE, signal
        signal(SIGPIPE, SIG_DFL)
    except (ImportError, AttributeError):
        pass

    ap = argparse.ArgumentParser(
        description="Bit-parallel approximate-multiplier BLIF simulator"
    )
    ap.add_argument(
        "--appMult", required=True, help="path to approximate multiplier BLIF"
    )
    ap.add_argument(
        "-s", "--signed", action="store_true", help="treat as signed multiplier"
    )
    ap.add_argument(
        "--mode",
        choices=["short", "full"],
        default="full",
        # help="output format: 'short' matches .txt references in app_mults/ "
            #  "(default); 'full' matches the C++ main.cc output line set",
    )
    args = ap.parse_args()

    blocks = split_models(args.appMult)
    if not blocks:
        sys.exit(f"error: no .model found in {args.appMult}")

    n_models = len(blocks)
    top_name = blocks[0][0]
    if n_models > 1:
        print(
            f"Warning: The design has {n_models} root-level modules. "
            f"The first one ({top_name}) will be used."
        )

    registry = {}
    for name, text in blocks:
        registry[name] = parse_one_model(text)

    top = registry[top_name]
    n_pi = len(top.inputs)
    n_po = len(top.outputs)
    if not (n_pi <= 20 and n_pi % 2 == 0 and n_pi == n_po):
        sys.exit(
            f"error: expected n_pi == n_po, n_pi even, n_pi <= 20; "
            f"got n_pi={n_pi}, n_po={n_po}"
        )

    flat_nodes = flatten(top_name, registry)
    # ABC patches undriven nets with constant-0 drivers and prints a warning;
    # mirror that behaviour so byte-for-byte diff against the reference .txt
    # files stays clean.
    inject_const0_drivers(flat_nodes, top.inputs, top_name)
    po_arrs, n_frame = simulate(flat_nodes, top.inputs, top.outputs, n_pi)

    bit_width = n_pi // 2
    outp_unsigned = decode_outputs(po_arrs, n_frame)
    lut_unsigned = outp_unsigned.reshape(1 << bit_width, 1 << bit_width).T

    opA_idx = np.arange(1 << bit_width, dtype=np.int64)
    opB_idx = np.arange(1 << bit_width, dtype=np.int64)
    if args.signed:
        opA_grid = signed_extend(opA_idx, bit_width)
        opB_grid = signed_extend(opB_idx, bit_width)
        lut = signed_extend(lut_unsigned, n_po)
    else:
        opA_grid = opA_idx
        opB_grid = opB_idx
        lut = lut_unsigned.astype(np.int64)
    ref = np.outer(opA_grid, opB_grid).astype(np.int64)

    err = lut - ref
    er = float((lut != ref).mean())
    abs_err = np.abs(err).astype(np.int64)
    med = float(abs_err.mean())
    nmed = med / (1 << (2 * bit_width))
    mse = float((err.astype(np.float64) ** 2).mean())
    max_ed = int(abs_err.max())
    # MRED uses max(|ref|, 1) as the denominator so that ref==0 entries
    # contribute |err| (not |err|*1e9 as the C++ formula would). Verified
    # against the .txt references in app_mults/evo_selected/mult8u/.
    red = float(
        (abs_err.astype(np.float64) / np.maximum(np.abs(ref).astype(np.float64), 1.0)).mean()
    )
    err_mean = float(err.astype(np.float64).mean())
    err_var = float(err.astype(np.float64).var())

    out = []
    if args.mode == "full":
        out.append(
            "INFO: Signed approximate multiplier"
            if args.signed
            else "INFO: Unsigned approximate multiplier"
        )
    out.append(f"INFO: Error rate: {fmt_g(er)}")
    out.append(f"INFO: Mean error distance: {fmt_g(med)}")
    out.append(f"INFO: Normalized mean error distance: {fmt_g(nmed)}")
    out.append(f"INFO: Mean square error: {fmt_g(mse)}")
    if args.mode == "full":
        out.append(f"INFO: Max error distance: {max_ed}")
        out.append(f"INFO: Mean relative error distance: {fmt_g(red)}")
        out.append(f"INFO: Error (app_out - acc_out) mean: {fmt_g(err_mean)}")
        out.append(f"INFO: Error (app_out - acc_out) variance: {fmt_g(err_var)}")
    out.append("LUT for approximate multiplier:")
    sys.stdout.write("\n".join(out) + "\n")

    n_a = opA_grid.size
    n_b = opB_grid.size
    A_rep = np.repeat(opA_grid, n_b).tolist()
    B_til = np.tile(opB_grid, n_a).tolist()
    lut_flat = lut.flatten().tolist()
    sys.stdout.write(
        "\n".join(f"{a} {b} {p}" for a, b, p in zip(A_rep, B_til, lut_flat))
    )
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
