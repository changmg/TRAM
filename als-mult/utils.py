from typing import Optional
import os, subprocess, shutil


from mult_logicnet import LogicNet


# ------------- DOT Exporter -----------------
def _ensure_parent_dir(path: str):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def render_png_from_dot(dot_path: str, png_path: str, renderer: str = "dot", timeout: int = 30) -> bool:
    _ensure_parent_dir(png_path)
    renderer_exe = shutil.which(renderer)
    if not renderer_exe:
        print(f"{renderer} executable not found; skipping PNG export")
        return False
    try:
        subprocess.run(
            [renderer_exe, "-Tpng", dot_path, "-o", png_path],
            check=True,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
        print(f"{renderer} PNG written to {png_path}")
        return True
    except subprocess.TimeoutExpired:
        print(f"{renderer} rendering timed out after {timeout} seconds")
    except subprocess.CalledProcessError as error:
        stderr = ""
        if error.stderr:
            stderr = error.stderr.decode(errors="ignore")
        print(f"{renderer} rendering failed: {stderr[:80]}")
    return False


def write_dot(net: LogicNet, dot_path: str, png_path: Optional[str] = None, renderer: str = "dot", timeout: int = 30):
    dot_text = net.to_dot()
    _ensure_parent_dir(dot_path)
    with open(dot_path, "w") as f:
        f.write(dot_text)
    if png_path:
        render_png_from_dot(dot_path, png_path, renderer=renderer, timeout=timeout)
    print(f"DOT file written to {dot_path}")


def write_dot_and_png(net: LogicNet, dot_path: str, png_path: str, renderer: str = "dot", timeout: int = 30):
    write_dot(net, dot_path, png_path=png_path, renderer=renderer, timeout=timeout)


# ------------- Verilog Exporter -----------------
def write_verilog(net: LogicNet, nbits: int, filename: str):
    """
    Export the given LogicNet to a Verilog file.
    """
    lines = []
    # Define HA and FA modules first
    lines.append("""
module HA(input x, input y, output s, output co);
  assign s  = x ^ y;
  assign co = x & y;
endmodule

module FA(input x, input y, input z, output s, output co);
  assign s  = x ^ y ^ z;
  assign co = (x & y) | (x & z) | (y & z);
endmodule
""")

    # Define main multiplier module
    lines.append(f"// Auto-generated from LogicNet")
    lines.append(f"module Mult_{nbits}_{nbits} (")
    lines.append(f"    input  [{nbits-1}:0] IN1,")
    lines.append(f"    input  [{nbits-1}:0] IN2,")
    lines.append(f"    output [{2*nbits-1}:0] Out")
    lines.append(f");")
    lines.append("")

    # Gather internal wires (exclude IN, OUT, CONST)
    wires = set()
    for n in net.nodes.values():
        if n.kind in ("IN", "OUT", "CONST"):
            continue
        for out_sig in n.outputs:
            if out_sig.startswith("NULL"):
                continue
            wires.add(out_sig)
    if wires:
        lines.append("wire " + ", ".join(sorted(wires)) + ";")
        lines.append("")
    lines.append("wire CONST0_OUT;")
    lines.append("wire CONST1_OUT;")
    lines.append("assign CONST0_OUT = 1'b0;")
    lines.append("assign CONST1_OUT = 1'b1;")
    lines.append("")

    # Emit node instances
    for n in net.nodes.values():
        if n.kind in ("IN", "CONST"):
            continue
        elif n.kind == "AND":
            # two-input AND
            in0, in1 = n.inputs
            out0 = n.outputs[0]
            if not out0.startswith("NULL"):
                lines.append(f"assign {out0} = {in0} & {in1};")
        elif n.kind == "HA":
            if len(n.inputs) >= 2 and len(n.outputs) >= 2:
                in0, in1 = n.inputs[:2]
                s, co = n.outputs[:2]
                if s == 'NULL': # just make the output empty
                    s = ''
                if co == 'NULL':
                    co = ''
                lines.append(f"HA {n.name} (.x({in0}), .y({in1}), .s({s}), .co({co}));")
        elif n.kind == "FA":
            if len(n.inputs) >= 3 and len(n.outputs) >= 2:
                in0, in1, in2 = n.inputs[:3]
                s, co = n.outputs[:2]
                if s == 'NULL': # just make the output empty
                    s = ''
                if co == 'NULL':
                    co = ''
                lines.append(f"FA {n.name} (.x({in0}), .y({in1}), .z({in2}), .s({s}), .co({co}));")
        elif n.kind == "OUT":
            assert len(n.inputs) == 1
            lines.append(f"assign {n.name} = {n.inputs[0]};")

    lines.append("")

    # # Assign outputs z[k] from z{k}
    # for k in range(2 * nbits):
    #     sig = f"Z{k}"
    #     lines.append(f"assign z[{k}] = {sig};")

    lines.append("endmodule")
    verilog_code = "\n".join(lines)

    with open(filename, "w") as f:
        f.write(verilog_code)

    print(f"Verilog file written to: {filename}")