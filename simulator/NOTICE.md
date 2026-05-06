# Third-Party Components

## blifparser

The directory [`blifparser/`](blifparser/) is vendored verbatim from
**mario33881/blifparser** — a Python BLIF parser released under the MIT
license.

- Upstream: https://github.com/mario33881/blifparser
- Vendored at commit: `1aa73ef248399d4bcc7840169435c412a99288b3`
- License: MIT (see [`blifparser/LICENSE.txt`](blifparser/LICENSE.txt))

We use `blifparser` only to parse one `.model` block at a time. Multi-model
handling (one BLIF file containing several `.model` sections) and recursive
`.subckt` flattening are implemented in [`sim.py`](sim.py) on top of
`blifparser`'s per-model objects.

Acknowledgement: thank you to [@mario33881](https://github.com/mario33881)
for releasing `blifparser` openly.
