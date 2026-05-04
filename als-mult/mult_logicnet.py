from dataclasses import dataclass, field
from typing import List, Dict, Optional
import torch

# ------------- Core classes -----------------
@dataclass
class Node:
    name: str
    kind: str  # "AND", "HA", "FA", "IN", "OUT"
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)

class LogicNet:
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.producer: Dict[str, str] = {}
        self.consumers: Dict[str, List[str]] = {}
        self.node_positions: Dict[str, tuple] = {}

    def add_node(self, node: Node):
        if node.name in self.nodes:
            raise ValueError(f"Node {node.name} already exists")
        self.nodes[node.name] = node
        for s in node.outputs:
            if s in self.producer:
                raise ValueError(f"Signal {s} already has a producer {self.producer[s]}")
            self.producer[s] = node.name
        for s in node.inputs:
            self.consumers.setdefault(s, [])

    def connect(self, src_signal: str, dst_node_name: str):
        self.consumers.setdefault(src_signal, [])
        if dst_node_name not in self.consumers[src_signal]:
            self.consumers[src_signal].append(dst_node_name)

    def to_dot(self) -> str:
        # Calculate layers for each node based on topological order
        layers = self._calculate_layers()
        
        lines = ["digraph ArrayMultiplier {",
                 '  rankdir=TB;',  # Top to bottom to show layers vertically
                 '  node [shape=record, fontsize=10];',
                 '  edge [color="#00000033", penwidth=1];',
                 '  graph [splines=polyline];',
                 '']

        kind_styles = {
            "AND": 'shape=box,style="rounded,filled",fillcolor="#e6f2ff"',
            "HA":  'shape=box,style="rounded,filled",fillcolor="#e8ffe6"',
            "FA":  'shape=box,style="rounded,filled",fillcolor="#fff3e6"',
            "IN":  'shape=oval,style="filled",fillcolor="#f0f0f0"',
            "OUT": 'shape=oval,style="filled",fillcolor="#f0f0f0"',
            "CONST": 'shape=diamond,style="filled",fillcolor="#d9d9d9"',
        }

        def parse_column(name: str) -> Optional[int]:
            if "_c" in name:
                tail = name.split("_c", 1)[1]
                digits = ""
                for ch in tail:
                    if ch.isdigit():
                        digits += ch
                    else:
                        break
                if digits:
                    return int(digits)
            if name.startswith("Out[") and name.endswith("]"):
                try:
                    return int(name[name.find("[")+1:name.find("]")])
                except ValueError:
                    return 0
            return 0
        
        # Group nodes by layer
        nodes_by_layer: Dict[int, List[Node]] = {}
        for n in self.nodes.values():
            layer = layers.get(n.name, -1)
            if layer not in nodes_by_layer:
                nodes_by_layer[layer] = []
            nodes_by_layer[layer].append(n)
        
        # For PP nodes, gather grid position
        pp_nodes: Dict[tuple, Node] = {}  # (row, col) -> node
        max_pp_row = 0
        max_pp_col = 0
        spacing_x = 1.5
        spacing_y = 1.5
        signal_alias: Dict[str, str] = {}
        
        column_stack: Dict[int, int] = {}

        for n in self.nodes.values():
            if n.kind == "AND" and "PP" in n.name:
                try:
                    pp_part = n.name.split('(')[0]
                    parts = pp_part.replace("PP", "").split("_")
                    row, col = int(parts[0]), int(parts[1])
                    pp_nodes[(row, col)] = n
                    max_pp_row = max(max_pp_row, row)
                    max_pp_col = max(max_pp_col, col)
                except ValueError:
                    continue
        pp_bottom_y = -max_pp_row * spacing_y if pp_nodes else 0
        # Pre-populate column heights from stored positions (needs pp_bottom_y)
        for name, pos in self.node_positions.items():
            node = self.nodes.get(name)
            if not node:
                continue
            if node.kind not in ["HA", "FA", "OUT"]:
                continue
            col = parse_column(name)
            if col is None:
                continue
            _, y = pos
            height = int(round((pp_bottom_y - y) / spacing_y - 1))
            column_stack[col] = max(column_stack.get(col, 0), height + 1)
        
        def parse_column(name: str) -> int:
            if "_c" in name:
                tail = name.split("_c", 1)[1]
                digits = ""
                for ch in tail:
                    if ch.isdigit():
                        digits += ch
                    else:
                        break
                if digits:
                    return int(digits)
            if name.startswith("Out[") and name.endswith("]"):
                try:
                    return int(name[name.find("[")+1:name.find("]")])
                except ValueError:
                    return 0
            return 0
        
        def node_entry(name: str, style: str, label: str, x: float = None, y: float = None):
            if x is not None and y is not None:
                return f'  "{name}" [{style},label="{label}",pos="{x},{y}!",pin=true];'
            return f'  "{name}" [{style},label="{label}"];'
        
        def point_node(name: str, x: float, y: float):
            return f'  "{name}" [shape=point,width=0.05,height=0.05,label="",pos="{x},{y}!",pin=true];'
        
        # Emit nodes with explicit placement
        for n in self.nodes.values():
            style = kind_styles.get(n.kind, "")
            layer = layers.get(n.name, -1)
            label = f"{n.name}\\n{n.kind}\\nL{layer}"
            pos_x = pos_y = None
            
            if n.kind == "IN":
                if n.name.startswith("A"):
                    try:
                        idx = int(n.name[1:])
                        pos_x = idx * spacing_x
                        pos_y = spacing_y * 3
                    except ValueError:
                        pass
                elif n.name.startswith("B"):
                    try:
                        idx = int(n.name[1:])
                        pos_x = idx * spacing_x
                        pos_y = spacing_y * 2
                    except ValueError:
                        pass
            elif n.kind == "CONST":
                pos_x = -spacing_x
                pos_y = spacing_y * 3.5
            elif n.kind == "AND" and "PP" in n.name:
                try:
                    pp_part = n.name.split('(')[0]
                    parts = pp_part.replace("PP", "").split("_")
                    row, col = int(parts[0]), int(parts[1])
                    label = f"{n.name}\\n{n.kind}\\nL{layer} (r{row},c{col})"
                    pos_x = col * spacing_x
                    pos_y = -row * spacing_y
                except ValueError:
                    pass
            elif n.kind in ["HA", "FA"]:
                col = parse_column(n.name)
                label = f"{n.name}\\n{n.kind}\\nL{layer} c{col}"
                stored = self.node_positions.get(n.name)
                if stored:
                    pos_x, pos_y = stored
                elif layer >= 2:
                    pos_x = col * spacing_x
                    height = column_stack.get(col, 0)
                    pos_y = pp_bottom_y - spacing_y * (height + 1)
                    column_stack[col] = height + 1
                    self.node_positions[n.name] = (pos_x, pos_y)
            elif n.kind == "OUT":
                col = parse_column(n.name)
                stored = self.node_positions.get(n.name)
                if stored:
                    pos_x, pos_y = stored
                elif layer >= 2:
                    pos_x = col * spacing_x
                    height = column_stack.get(col, 0)
                    pos_y = pp_bottom_y - spacing_y * (height + 1)
                    column_stack[col] = height + 1
                    self.node_positions[n.name] = (pos_x, pos_y)
            else:
                col = parse_column(n.name)
                stored = self.node_positions.get(n.name)
                if stored:
                    pos_x, pos_y = stored
                elif layer >= 2:
                    pos_x = col * spacing_x
                    height = column_stack.get(col, 0)
                    pos_y = pp_bottom_y - spacing_y * (height + 1)
                    column_stack[col] = height + 1
                    self.node_positions[n.name] = (pos_x, pos_y)
            
            if pos_x is not None and pos_y is not None and n.name not in self.node_positions:
                self.node_positions[n.name] = (pos_x, pos_y)

            lines.append(node_entry(n.name, style, label, pos_x, pos_y))
            
            # Add ports for adders to enforce horizontal/vertical routing
            if n.kind in ["HA", "FA"] and pos_x is not None and pos_y is not None:
                outputs = [sig for sig in n.outputs if not sig.startswith("NULL")]
                s_sig = outputs[0] if outputs else None
                co_sig = outputs[1] if len(outputs) > 1 else None
                if s_sig:
                    sum_port = f"{n.name}_SUMPORT"
                    sum_y = pos_y - spacing_y * 0.45
                    lines.append(point_node(sum_port, pos_x, sum_y))
                    lines.append(f'  "{n.name}" -> "{sum_port}" [dir=none];')
                    signal_alias[s_sig] = sum_port
                if co_sig:
                    car_port = f"{n.name}_CARPORT"
                    car_x = pos_x + spacing_x * 0.45
                    lines.append(point_node(car_port, car_x, pos_y))
                    lines.append(f'  "{n.name}" -> "{car_port}" [dir=none];')
                    signal_alias[co_sig] = car_port

        lines.append('')
        
        # Add rank constraints to force nodes in same layer to be at same level
        for layer in sorted(nodes_by_layer.keys()):
            if layer >= 0 and nodes_by_layer[layer]:
                node_names = [f'"{n.name}"' for n in nodes_by_layer[layer]]
                lines.append(f'  {{ rank=same; {" ".join(node_names)}; }}')
        
        lines.append('')
        
        # Create subgraph for partial product grid to enforce 2D layout
        if pp_nodes:
            # Find grid dimensions
            max_row = max(pos[0] for pos in pp_nodes.keys())
            max_col = max(pos[1] for pos in pp_nodes.keys())
            
            lines.append('  // Partial product 2D grid layout')
            # For each row, create invisible edges to enforce left-to-right order
            for row in range(max_row + 1):
                row_nodes = []
                for col in range(max_col + 1):
                    if (row, col) in pp_nodes:
                        row_nodes.append(f'"{pp_nodes[(row, col)].name}"')
                
                if len(row_nodes) > 1:
                    # Create invisible edges to maintain order
                    for i in range(len(row_nodes) - 1):
                        lines.append(f'  {row_nodes[i]} -> {row_nodes[i+1]} [style=invis, weight=10];')
            
            lines.append('')

        # Emit edges
        for sig, prod in self.producer.items():
            source = signal_alias.get(sig, prod)
            consumers = self.consumers.get(sig, [])
            for dst in consumers:
                lines.append(f'  "{source}" -> "{dst}" [label="{sig}", fontsize=8];')

        lines.append("}")
        return "\n".join(lines)
    
    def _calculate_layers(self) -> Dict[str, int]:
        """
        Calculate the layer (level) of each node based on maximum distance from inputs.
        This naturally groups nodes at the same depth in the Wallace tree together.
        - Layer 0: Inputs
        - Layer 1: Partial products (AND gates)
        - Layer 2+: Reduction tree (adders grouped by depth)
        """
        layers: Dict[str, int] = {}
        
        # Layer 0: All input nodes
        for name, node in self.nodes.items():
            if node.kind == "IN" or node.kind == "CONST":
                layers[name] = 0
        
        # Layer 1: All partial products (AND gates)
        for name, node in self.nodes.items():
            if node.kind == "AND":
                layers[name] = 1
        
        # For remaining nodes, compute layer based on maximum input layer + 1
        # This groups nodes at the same depth together
        changed = True
        while changed:
            changed = False
            for name, node in self.nodes.items():
                if name in layers:
                    continue
                
                # Find the maximum layer of all input producers
                max_input_layer = -1
                all_inputs_assigned = True
                
                for inp_sig in node.inputs:
                    if inp_sig in self.producer:
                        prod_name = self.producer[inp_sig]
                        if prod_name not in layers:
                            all_inputs_assigned = False
                            break
                        max_input_layer = max(max_input_layer, layers[prod_name])
                
                if all_inputs_assigned and max_input_layer >= 0:
                    layers[name] = max_input_layer + 1
                    changed = True
        
        return layers


# ------------- Build Array Multiplier -----------------
def build_array_multiplier(nbits: int = 4) -> LogicNet:
    net = LogicNet()

    for i in range(nbits):
        net.add_node(Node(name=f"A{i}", kind="IN", outputs=[f"IN1[{i}]"]))
        net.add_node(Node(name=f"B{i}", kind="IN", outputs=[f"IN2[{i}]"]))

    p = [[f"p{i}_{j}" for j in range(nbits)] for i in range(nbits)]
    for i in range(nbits):
        for j in range(nbits):
            and_name = f"PP{i}_{j}(c{i+j})"
            node = Node(name=and_name, kind="AND", inputs=[f"IN1[{i}]", f"IN2[{j}]"], outputs=[p[i][j]])
            net.add_node(node)
            net.connect(f"IN1[{i}]", and_name)
            net.connect(f"IN2[{j}]", and_name)

    columns: Dict[int, List[str]] = {c: [] for c in range(2 * nbits)}
    for i in range(nbits):
        for j in range(nbits):
            columns[i + j].append(p[i][j])

    # Wallace tree reduction: process all columns in parallel at each stage
    fa_count = 0
    ha_count = 0
    stage = 0
    
    max_height = max(len(columns[c]) for c in range(2 * nbits))
    while max_height > 2:
        stage += 1
        new_columns: Dict[int, List[str]] = {c: [] for c in range(2 * nbits)}
        
        # In this stage, reduce all columns simultaneously
        for c in range(2 * nbits):
            remaining = columns[c][:]
            
            # Add FAs to reduce groups of 3 bits
            while len(remaining) >= 3:
                x, y, z = remaining.pop(0), remaining.pop(0), remaining.pop(0)
                s = f"s_c{c}_fa{fa_count}"
                co = f"co_c{c}_fa{fa_count}"
                fa_name = f"FA_c{c}_{fa_count}"
                net.add_node(Node(name=fa_name, kind="FA", inputs=[x, y, z], outputs=[s, co]))
                for inp in [x, y, z]:
                    net.connect(inp, fa_name)
                new_columns[c].append(s)
                if c + 1 < 2 * nbits:
                    new_columns[c + 1].append(co)
                fa_count += 1
            
            # Remaining bits pass through to next stage
            new_columns[c].extend(remaining)
        
        columns = new_columns
        max_height = max(len(columns[c]) for c in range(2 * nbits))
        
        # Safety check
        if stage > 20:
            raise RuntimeError(f"Too many stages in Wallace tree reduction")
    
    # Final stage: use HAs to reduce remaining 2-bit columns
    ha_stage = 0
    while max(len(columns[c]) for c in range(2 * nbits)) > 1:
        ha_stage += 1
        if ha_stage > 20:
            raise RuntimeError(f"Too many HA stages, columns: {[len(columns[c]) for c in range(2 * nbits)]}")
        
        any_reduced = False
        for c in range(2 * nbits):
            if len(columns[c]) == 2:
                x, y = columns[c].pop(0), columns[c].pop(0)
                s = f"s_c{c}_ha{ha_count}"
                co = f"co_c{c}_ha{ha_count}"
                ha_name = f"HA_c{c}_{ha_count}"
                net.add_node(Node(name=ha_name, kind="HA", inputs=[x, y], outputs=[s, co]))
                net.connect(x, ha_name)
                net.connect(y, ha_name)
                columns[c].append(s)
                if c + 1 < 2 * nbits:
                    columns[c + 1].append(co)
                ha_count += 1
                any_reduced = True
        
        if not any_reduced:
            # No progress made, we're stuck - columns might have > 2 bits
            # Use FAs to reduce
            for c in range(2 * nbits):
                if len(columns[c]) >= 3:
                    x, y, z = columns[c].pop(0), columns[c].pop(0), columns[c].pop(0)
                    s = f"s_c{c}_fa{fa_count}"
                    co = f"co_c{c}_fa{fa_count}"
                    fa_name = f"FA_c{c}_{fa_count}"
                    net.add_node(Node(name=fa_name, kind="FA", inputs=[x, y, z], outputs=[s, co]))
                    for inp in [x, y, z]:
                        net.connect(inp, fa_name)
                    columns[c].append(s)
                    if c + 1 < 2 * nbits:
                        columns[c + 1].append(co)
                    fa_count += 1

    product_bits: List[str] = []
    for c in range(0, 2 * nbits):
        if len(columns[c]) != 1:
            raise RuntimeError(f"Column {c} did not reduce to one bit, has {len(columns[c])} bits")
        product_bits.append(columns[c][0])
    
    print(f'Total FAs: {fa_count}, Total HAs: {ha_count}')

    for k, sig in enumerate(product_bits):
        out_name = f"Out[{k}]"
        net.add_node(Node(name=out_name, kind="OUT", inputs=[sig], outputs=[f"Out[{k}]"]))
        net.connect(sig, out_name)

    return net


# ------------- Modify circuit -----------------
@dataclass
class SignalModificationLog:
    target_signal: str
    const_value: int
    old_producer: str
    producer_node_name: str
    old_node_outputs: List[str]
    consumer_inputs: Dict[str, List[str]]
    consumer_list: List[str]
    const_node_added: bool
    const_node_name: str
    const_output_signal: str
    const_output_consumers_prev: Optional[List[str]]
    target_consumers_prev: Optional[List[str]]


def replace_target_signal_with_const(net: LogicNet, target_signal: str, const_value: int, record_changes: bool = False) -> Optional[SignalModificationLog]:
    """
    Replace the producer of the signal with a constant 0.
    """
    # Check if the signal exists
    if target_signal not in net.producer:
        raise KeyError(f"Signal '{target_signal}' not found in net.producer")

    # Find the original producer node
    old_producer = net.producer[target_signal]
    old_node = net.nodes[old_producer]

    # # Remove the signal from the old node's outputs
    # if target_signal in old_node.outputs:
    #     old_node.outputs.remove(target_signal)
    original_outputs = old_node.outputs[:]
    if target_signal in old_node.outputs:
        old_node.outputs = [s if s != target_signal else "NULL" for s in old_node.outputs]
    # print(f"Old node {old_node.name}'s outputs after removal: {old_node.outputs}")

    del net.producer[target_signal]

    consumer_list = net.consumers.get(target_signal, [])
    target_consumers_prev = consumer_list[:] if target_signal in net.consumers else None

    # Create const0 node if not already present
    const_name = "CONST0" if const_value == 0 else "CONST1"
    output_signal = "CONST0_OUT" if const_value == 0 else "CONST1_OUT"
    const_node_added = False
    if const_name not in net.nodes:
        const_node = Node(name=const_name, kind="CONST", inputs=[], outputs=[output_signal])
        net.add_node(const_node)
        const_node_added = True

    consumer_inputs: Dict[str, List[str]] = {}
    for consumer_name in consumer_list:
        consumer_node = net.nodes[consumer_name]
        consumer_inputs[consumer_name] = consumer_node.inputs[:]
        consumer_node.inputs = [output_signal if x == target_signal else x for x in consumer_node.inputs]

    prev_const_consumers = net.consumers.get(output_signal)
    prev_const_consumers_copy = prev_const_consumers[:] if prev_const_consumers is not None else None
    net.consumers.setdefault(output_signal, [])
    for c in consumer_list:
        if c not in net.consumers[output_signal]:
            net.consumers[output_signal].append(c)

    if target_signal in net.consumers:
        del net.consumers[target_signal]

    if record_changes:
        return SignalModificationLog(
            target_signal=target_signal,
            const_value=const_value,
            old_producer=old_producer,
            producer_node_name=old_node.name,
            old_node_outputs=original_outputs,
            consumer_inputs=consumer_inputs,
            consumer_list=consumer_list[:],
            const_node_added=const_node_added,
            const_node_name=const_name,
            const_output_signal=output_signal,
            const_output_consumers_prev=prev_const_consumers_copy,
            target_consumers_prev=target_consumers_prev,
        )
    return None

def restore_signal(net: LogicNet, log: SignalModificationLog):
    node = net.nodes[log.producer_node_name]
    node.outputs = log.old_node_outputs
    net.producer[log.target_signal] = log.old_producer

    for consumer_name, inputs in log.consumer_inputs.items():
        net.nodes[consumer_name].inputs = inputs

    if log.target_consumers_prev is None:
        net.consumers.pop(log.target_signal, None)
    else:
        net.consumers[log.target_signal] = log.target_consumers_prev[:]

    if log.const_output_consumers_prev is None:
        net.consumers.pop(log.const_output_signal, None)
    else:
        net.consumers[log.const_output_signal] = log.const_output_consumers_prev[:]

    if log.const_node_added and log.const_node_name in net.nodes:
        const_node = net.nodes.pop(log.const_node_name)
        for sig in const_node.outputs:
            if net.producer.get(sig) == const_node.name:
                del net.producer[sig]
            net.consumers.pop(sig, None)


def replace_target_signal_with_const0(net: LogicNet, target_signal: str):
    """
    Replace the producer of the signal with constant 0.
    Wrapper function for backward compatibility.
    """
    replace_target_signal_with_const(net, target_signal, 0)


# ------------- Topological order -----------------
# def topo_order(net: LogicNet) -> List[str]:
#     preds: Dict[str, set] = {name: set() for name in net.nodes}
#     succs: Dict[str, List[str]] = {name: [] for name in net.nodes}
#     for node in net.nodes.values():
#         for sig in node.inputs:
#             prod = net.producer.get(sig)
#             if prod is not None:
#                 preds[node.name].add(prod)
#                 succs[prod].append(node.name)
#     order: List[str] = []
#     ready = [n for n, ps in preds.items() if len(ps) == 0]
#     while ready:
#         u = ready.pop(0)
#         order.append(u)
#         for v in succs[u]:
#             preds[v].discard(u)
#             if len(preds[v]) == 0:
#                 ready.append(v)
#     if len(order) != len(net.nodes):
#         print(f'order: {order}')
#         print(f'net nodes: {list(net.nodes.keys())}')
#         print(f'len(order): {len(order)}, len(net.nodes): {len(net.nodes)}')
#         raise RuntimeError("Cycle detected or missing producers in the graph")
#     return order
def topo_order(net: LogicNet) -> List[str]:
    preds: Dict[str, set] = {name: set() for name in net.nodes}
    succs: Dict[str, set] = {name: set() for name in net.nodes}

    for node in net.nodes.values():
        for sig in node.inputs:
            prod = net.producer.get(sig)
            if prod is not None:
                preds[node.name].add(prod)       # distinct predecessors only
                succs[prod].add(node.name)       # distinct successors only

    order: List[str] = []
    ready: List[str] = [n for n, ps in preds.items() if len(ps) == 0]
    enqueued = set(ready)                        # avoid duplicate enqueues

    while ready:
        u = ready.pop(0)
        order.append(u)
        for v in succs[u]:
            if u in preds[v]:
                preds[v].discard(u)
            if len(preds[v]) == 0 and v not in enqueued:
                ready.append(v)
                enqueued.add(v)

    if len(order) != len(net.nodes):
        print(f'order: {order}')
        print(f'net nodes: {list(net.nodes.keys())}')
        print(f'len(order): {len(order)}, len(net.nodes): {len(net.nodes)}')
        raise RuntimeError("Cycle detected or missing producers in the graph")
    return order


# ------------- Simulator -----------------
def enumerate_patterns_bool(nbits: int) -> torch.Tensor:
    num_inputs = 2 * nbits
    N = 1 << num_inputs
    pat = torch.zeros((N, num_inputs), dtype=torch.bool)
    for idx in range(N):
        for i in range(nbits):
            pat[idx, i] = bool((idx >> i) & 1)                 # A_i
            pat[idx, nbits + i] = bool((idx >> (nbits + i)) & 1)  # B_i
    return pat


def bits_to_uint(bits: List[torch.Tensor]) -> torch.Tensor:
    acc = torch.zeros(bits[0].shape[0], dtype=torch.long)
    for i, b in enumerate(bits):
        acc = acc + b.to(torch.long) * (1 << i)
    return acc


def simulate_multiplier(net: LogicNet, nbits: int, patterns: Optional[torch.Tensor] = None, A_val: Optional[torch.Tensor] = None, B_val: Optional[torch.Tensor] = None):
    order = topo_order(net)
    if patterns is None:
        patterns = enumerate_patterns_bool(nbits)
    if patterns.shape[1] != 2 * nbits:
        raise ValueError(f"patterns should have shape (_, {2 * nbits}), got {patterns.shape}")
    N = patterns.shape[0]
    values: Dict[str, torch.Tensor] = {}
    for i in range(nbits):
        values[f"IN1[{i}]"] = patterns[:, i]
        values[f"IN2[{i}]"] = patterns[:, nbits + i]
    for name in order:
        # print(f"Simulating node: {name}")
        node = net.nodes[name]
        if node.kind == "IN":
            for s in node.outputs:
                values.setdefault(s, torch.zeros(N, dtype=torch.bool))
        elif node.kind == "AND":
            a, b = values[node.inputs[0]], values[node.inputs[1]]
            if node.outputs[0] != 'NULL':
                values[node.outputs[0]] = a & b
            # else:
            #     print(f"Output of AND node {name} is 'NULL', skipping simulation.")
        elif node.kind == "HA":
            x, y = values[node.inputs[0]], values[node.inputs[1]]
            if node.outputs[0] != 'NULL':
                s = x ^ y
                values[node.outputs[0]] = s
            # else:
            #     print(f"Sum output of HA node {name} is 'NULL', skipping simulation.")
            if node.outputs[1] != 'NULL':
                co = x & y
                values[node.outputs[1]] = co
            # else:
            #     print(f"Carry output of HA node {name} is 'NULL', skipping simulation.")
        elif node.kind == "FA":
            x, y, z = values[node.inputs[0]], values[node.inputs[1]], values[node.inputs[2]]
            if node.outputs[0] != 'NULL':
                s = x ^ y ^ z
                values[node.outputs[0]] = s
            # else:
            #     print(f"Sum output of FA node {name} is 'NULL', skipping simulation.")
            if node.outputs[1] != 'NULL':
                co = (x & y) | (x & z) | (y & z)
                values[node.outputs[1]] = co
            # else:
            #     print(f"Carry output of FA node {name} is 'NULL', skipping simulation.")
        elif node.kind == "OUT":
            src = values[node.inputs[0]]
            values[node.outputs[0]] = src
        elif node.kind == "CONST":
            if node.name == "CONST0":
                for s in node.outputs:
                    values[s] = torch.zeros(N, dtype=torch.bool)
            elif node.name == "CONST1":
                for s in node.outputs:
                    values[s] = torch.ones(N, dtype=torch.bool)
            else:
                raise RuntimeError(f"Unknown const node: {node.name}")
        else:
            raise RuntimeError(f"Unknown node kind: {node.kind}")
    z_bits = [values[f"Out[{k}]"] for k in range(2 * nbits)]
    if A_val is None:
        A_val = bits_to_uint([values[f"IN1[{i}]"] for i in range(nbits)])
    if B_val is None:
        B_val = bits_to_uint([values[f"IN2[{i}]"] for i in range(nbits)])
    Z_val = bits_to_uint(z_bits)
    ref_val = A_val * B_val
    return patterns, A_val, B_val, Z_val, ref_val