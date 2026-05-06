import torch 
import argparse
import os
import time
from typing import List
from tqdm import tqdm

from mult_logicnet import build_array_multiplier, simulate_multiplier, replace_target_signal_with_const, restore_signal, SignalModificationLog
from utils import write_dot_and_png, write_verilog


def get_bit(x: torch.Tensor, mask: int) -> torch.Tensor:
    return torch.bitwise_and(x.int(), mask)


def get_ith_bit(x: torch.Tensor, i: int) -> torch.Tensor:
    return get_bit(x, 1 << i)


def extract_compressors_in_column(net, column_id: int):
    compressors = []
    for node in net.nodes.values():
        if node.kind in ["HA", "FA"] and f"_c{column_id}_" in node.name:
            compressors.append(node)
    return compressors


def run_simulation(net, nbits, use_cached_inputs, patterns=None, A_val=None, B_val=None):
    if use_cached_inputs and patterns is not None and A_val is not None and B_val is not None:
        return simulate_multiplier(net, nbits, patterns=patterns, A_val=A_val, B_val=B_val)
    return simulate_multiplier(net, nbits)


def try_override(
    net,
    signal_batch: List[str],
    const_value: int,
    label: str,
    curr_mse: float,
    nbits: int,
    Y_target: torch.Tensor,
    patterns,
    A_val,
    B_val,
    use_cached_inputs: bool,
):
    logs: List[SignalModificationLog] = []
    for sig in signal_batch:
        log = replace_target_signal_with_const(net, sig, const_value=const_value, record_changes=True)
        logs.append(log)
    _, _, _, Y_temp, _ = run_simulation(
        net, nbits, use_cached_inputs, patterns=patterns, A_val=A_val, B_val=B_val
    )
    mse = torch.mean((Y_temp.to(torch.float32) - Y_target) ** 2).item()
    if mse < curr_mse:
        if len(signal_batch) == 1:
            print(f'  signal {signal_batch[0]}: const{const_value} accepted with MSE = {mse:.4f}')
        else:
            print(f'  batch {label}: const{const_value} accepted (size={len(signal_batch)}) with MSE = {mse:.4f}')
        return True, mse
    for log in reversed(logs):
        restore_signal(net, log)
    return False, curr_mse


def process_signals(
    signals: List[str],
    const_value: int,
    kind: str,
    batch_size: int,
    net,
    nbits: int,
    Y_target: torch.Tensor,
    patterns,
    A_val,
    B_val,
    use_cached_inputs: bool,
    curr_mse: float,
):
    idx = 0
    while idx < len(signals):
        group = signals[idx : idx + max(1, batch_size)]
        batch_label = f'{kind}[{idx}:{idx + len(group)}]'
        improved = False
        updated_mse = curr_mse
        if batch_size > 1:
            improved, updated_mse = try_override(
                net,
                group,
                const_value,
                batch_label,
                curr_mse,
                nbits,
                Y_target,
                patterns,
                A_val,
                B_val,
                use_cached_inputs,
            )
        if not improved:
            for sig in group:
                _, updated_mse = try_override(
                    net,
                    [sig],
                    const_value,
                    f'{kind}:{sig}',
                    updated_mse,
                    nbits,
                    Y_target,
                    patterns,
                    A_val,
                    B_val,
                    use_cached_inputs,
                )
        curr_mse = updated_mse
        idx += len(group)
    return curr_mse


def main(
    nbits: int,
    target_gamma: List[float],
    out_verilog: str = None,
    batch_size: int = 4,
    use_cached_simulation: bool = False,
):
    cpu_start = time.process_time()

    # Pre-process gamma values
    num_max_discard_cols = nbits * 2 - 1
    assert len(target_gamma) <= num_max_discard_cols
    if len(target_gamma) < num_max_discard_cols:
        target_gamma = target_gamma + [0.0] * (num_max_discard_cols - len(target_gamma))
    print(f"Target gamma values for {num_max_discard_cols} columns: " + ", ".join([f"{lmbd:.8f}" for lmbd in target_gamma]))

    # Build and simulate the accurate array multiplier
    print(f"Building {nbits}-bit array multiplier...")
    net = build_array_multiplier(nbits)
    # Drop the accurate reference Verilog (Mult_<nbits>_<nbits>.v) next to the
    # caller-supplied out_verilog so all artifacts land in the same folder;
    # fall back to tmp/ for callers that don't pass out_verilog.
    output_dir = (os.path.dirname(out_verilog) if out_verilog else "") or "tmp"
    os.makedirs(output_dir, exist_ok=True)
    dot_file = os.path.join(output_dir, f"Mult_{nbits}_{nbits}.dot")
    output_png = os.path.join(output_dir, f"Mult_{nbits}_{nbits}_2dgrid.png")
    # write_dot_and_png(net, dot_file, output_png, renderer="neato")
    
    patterns, A_val, B_val, Y_val, _ = simulate_multiplier(net, nbits)
    Y_acc = A_val * B_val
    ok = (Y_val == Y_acc)
    print(f"Simulation {'passed' if ok.all() else 'failed'} for {patterns.shape[0]} patterns.")
    write_verilog(net=net, nbits=nbits, filename=os.path.join(output_dir, f"Mult_{nbits}_{nbits}.v"))

    # For each column c, obtain S_c = \sum_{i+j=c} A_i * B_j
    A_i = [get_ith_bit(A_val, i) for i in range(nbits)] # with weight 2^i, A_i[i] shape = (num_patterns,)
    B_i = [get_ith_bit(B_val, i) for i in range(nbits)] # with weight 2^i, B_i[i] shape = (num_patterns,)
    S_c = [torch.zeros_like(A_val) for _ in range(num_max_discard_cols)] # S_c[c] shape = (num_patterns,)
    # print(f"A_i = [" + "\n".join([f"{A_i[i]}" for i in range(nbits)]) + "]")
    # print(f"B_i = [" + "\n".join([f"{B_i[i]}" for i in range(nbits)]) + "]")
    Y_target = Y_acc.clone().to(torch.float32) # target output value
    for col in range(num_max_discard_cols): # for each column
        min_i = max(0, col - (nbits - 1))
        max_i = min(col + 1, nbits) # exclusive
        for i in range(min_i, max_i):
            j = col - i
            S_c[col] += A_i[i] * B_i[j]
        Y_target -= S_c[col].to(torch.float32) * target_gamma[col]
    # print(f"S_c = [" + "\n".join([f"{S_c[c]}" for c in range(num_max_discard_cols)]) + "]")
    # print(f'Y_acc = {Y_acc.tolist()}')
    # print(f"Y_target = {Y_target.tolist()}")
    print(f'MSE between accurate and target outputs: {torch.mean((Y_acc.to(torch.float32) - Y_target) ** 2).item():.4f}')

    # Extract compressors and their output signals in each column
    cand_sigs_in_cols = {}
    for col in range(num_max_discard_cols):
        compressors = extract_compressors_in_column(net=net, column_id=col)
        # print(f"Col{col}, {len(compressors)} compressors: " + ", ".join([c.name for c in compressors]))
        # For each compressor, collect its output signals as candidates for approximation
        cand_sigs_col_i = []
        for compressor in compressors:
            for out_sig in compressor.outputs:
                # print(f"  Compressor {compressor.name} output signal: {out_sig}")
                cand_sigs_col_i.append(out_sig)
        # print(f"Col{col}, {len(cand_sigs_col_i)} target signals: " + ", ".join(cand_sigs_col_i))
        cand_sigs_in_cols[col] = cand_sigs_col_i

    # Special handling for LSB column (column 0)
    replace_target_signal_with_const(net, "p0_0", const_value=0)

    # evaluation
    _, _, _, Y_curr, _ = run_simulation(
        net, nbits, use_cached_simulation, patterns=patterns, A_val=A_val, B_val=B_val
    )
    curr_mse = torch.mean((Y_curr.to(torch.float32) - Y_target) ** 2).item()
    print(f'Initial MSE after forcing p0_0=0: {curr_mse:.4f}')
    # Process column by column, start from simple cases (gamma=0.0 or 1.0)
    for col in range(num_max_discard_cols):
        if len(cand_sigs_in_cols[col]) == 0:
            # print(f'skipping column {col} (no target signals)')
            continue
        if target_gamma[col] == 0.0:
            print(f'skipping approximation in column {col} as target gamma is 0.0')
        elif target_gamma[col] == 1.0:
            print(f'zero all target signals in column {col} as target gamma is 1.0')
            for sig in cand_sigs_in_cols[col]:
                replace_target_signal_with_const(net, sig, const_value=0)
    
    # evaluation
    _, _, _, Y_curr, _ = run_simulation(
        net, nbits, use_cached_simulation, patterns=patterns, A_val=A_val, B_val=B_val
    )
    curr_mse = torch.mean((Y_curr.to(torch.float32) - Y_target) ** 2).item()
    print(f'Current MSE: {curr_mse: 4f}')
    # Process column by column, 
    for col in range(num_max_discard_cols):
        if len(cand_sigs_in_cols[col]) == 0:
            # print(f'skipping column {col} (no target signals)')
            continue
        if target_gamma[col] == 0.0 or target_gamma[col] == 1.0:
            continue
        assert 0.0 < target_gamma[col] < 1.0
        print(f'processing column {col} with target gamma = {target_gamma[col]:.2f}...')
        print(f'Col {col} candidate signals: ' + ", ".join(cand_sigs_in_cols[col]))
        # traverse each sum signal and decide its approximation
        sum_signals = [sig for sig in cand_sigs_in_cols[col] if sig.startswith('s_')]
        carry_signals = [sig for sig in cand_sigs_in_cols[col] if sig.startswith('co_c')]

        if sum_signals:
            print(f'  processing {len(sum_signals)} sum signals with batch_size={max(1, batch_size)}')
            curr_mse = process_signals(
                sum_signals,
                const_value=0,
                kind=f's_col{col}',
                batch_size=batch_size,
                net=net,
                nbits=nbits,
                Y_target=Y_target,
                patterns=patterns,
                A_val=A_val,
                B_val=B_val,
                use_cached_inputs=use_cached_simulation,
                curr_mse=curr_mse,
            )
        if carry_signals:
            print(f'  processing {len(carry_signals)} carry signals with batch_size={max(1, batch_size)}')
            curr_mse = process_signals(
                carry_signals,
                const_value=0,
                kind=f'co_col{col}',
                batch_size=batch_size,
                net=net,
                nbits=nbits,
                Y_target=Y_target,
                patterns=patterns,
                A_val=A_val,
                B_val=B_val,
                use_cached_inputs=use_cached_simulation,
                curr_mse=curr_mse,
            )

    # Evaluate final MSE after all approximations
    _, _, _, Y_final, _ = run_simulation(
        net, nbits, use_cached_simulation, patterns=patterns, A_val=A_val, B_val=B_val
    )
    final_mse = torch.mean((Y_final.to(torch.float32) - Y_target) ** 2).item()
    final_med = torch.mean(torch.abs(Y_final.to(torch.float32) - Y_target)).item()
    final_mred = torch.mean(torch.abs(Y_final.to(torch.float32) - Y_target) / (Y_target + 1e-6)).item()
    print(f'Final MSE after all approximations: {final_mse:.4f}')
    print(f'Final MED after all approximations: {final_med:.4f}')
    print(f'Final MRED after all approximations: {final_mred:.4f}')

    # Export approximate DOT + 2D grid viz
    approx_dot = os.path.join(output_dir, f"Mult_{nbits}_{nbits}_approx.dot")
    approx_png = os.path.join(output_dir, f"Mult_{nbits}_{nbits}_approx_2dgrid.png")
    # write_dot_and_png(net, approx_dot, approx_png, renderer="neato")

    # Export the final Verilog
    write_verilog(net=net, nbits=nbits, filename=out_verilog)

    cpu_elapsed = time.process_time() - cpu_start
    print(f'CPU time for finding best structure: {cpu_elapsed:.2f}s')


def parse_args():
    parser = argparse.ArgumentParser(description='running parameters', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--parse_file', default='', type=str, help='path to the log file containing target gamma values')

    return parser.parse_args()


# ------------- Main execution -----------------
if __name__ == "__main__":
    file = parse_args().parse_file
    assert file != ''
    cmd = f'cat {file} | grep "gamma      = \[" | tail -1'
    ret = os.popen(cmd).read()
    print(f'parsed line: {ret.strip()}')
    # example line: 2025-11-14 02:43:51,599 - layer_id = 20, gamma      = ['  1.000000', '  1.000000', '  1.000000', '  1.000000', '  0.636009', '  0.722808', '  0.693135', '  0.502075', '  0.023763']
    gamma_str = ret.split('gamma      = [')[1].split(']')[0]
    gamma = [val.strip().strip("'") for val in gamma_str.split(',')]
    target_gamma = [float(val) for val in gamma]
    # print(f'Parsed target gamma values: ' + ", ".join([f"{lmbd:.8f}" for lmbd in target_gamma]))
    out_verilog = file.replace('.log', '.v')
    print(f'Output verilog file: {out_verilog}')

    # target_gamma = [1.0, 1.0, 1.0, 1.0, 1.0, 0.0]
    # out_verilog = f'./tmp/Mult_8_8_peusdo_vt5.v'
    main(nbits=8, target_gamma=target_gamma, out_verilog=out_verilog, batch_size=1, use_cached_simulation=True)