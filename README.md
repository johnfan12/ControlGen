# ControlGen

ControlGen is a graph-based control-system dataset generator for structure-to-time-series tasks.
The current version targets layered large-scale MIMO control graphs with:

- `3x3` to `5x5` style multi-input multi-output systems by default
- `8-24` graph nodes sampled from reference, controller, actuator, process, sensor, disturbance, noise, and output layers
- Dense, rule-based tap placement across external inputs, controller outputs, actuator outputs, process outputs, sensor outputs, and primary outputs
- Linear node dynamics with local nonlinear wrappers such as saturation and rate limiting
- Multi-channel scenario generation for reference inputs, disturbances, and measurement noise

## What is implemented

- Graph DSL with `graph`, `node`, `edge`, `port`, and `tap`
- Low-level dynamic block DSL with `tf`, `gain`, `matgain`, `pid`, `delay`, `ss`, `series`, `parallel`, and `feedback`
- Layered MIMO graph generator with dense coupling and delayed feedback edges
- Graph-level simulation producing grouped signals:
  - `external_inputs`
  - `primary_outputs`
  - `tap_signals`
  - `disturbance_signals`
  - `noise_signals`
- Optional hidden-state teacher signals
- Dataset schema with graph structure, scenario, grouped signals, node models, and graph/system/channel metrics

## Quick start

```bash
python3 -m pip install -e .[dev]
python3 -m controlgen.cli --count 3 --seed 42 --min-nodes 12 --max-nodes 20 --min-io 4 --max-io 4
pytest -q
```

## Sample Graph DSL

```text
graph(
  topology_family='layered_mimo_graph',
  nodes=[
    node(
      id='ref_0',
      kind='reference_source',
      family='step',
      layer='reference',
      inputs=[],
      outputs=[port(name='y', direction='output', dimension=4, role='reference')],
      parameters={},
      metadata={}
    ),
    node(
      id='ctrl_0',
      kind='controller_bank',
      family='pi',
      layer='controller',
      inputs=[port(name='u', direction='input', dimension=2, role='controller_input')],
      outputs=[port(name='y', direction='output', dimension=2, role='controller_output')],
      parameters={},
      metadata={}
    )
  ],
  edges=[
    edge(
      id='edge_ref_ctrl_0',
      source='ref_0',
      source_port='y',
      target='ctrl_0',
      target_port='u',
      matrix=[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
      role='reference'
    )
  ],
  input_ports=[port(name='reference', direction='input', dimension=4, role='external_input')],
  output_ports=[port(name='system_output', direction='output', dimension=4, role='primary_output')],
  taps=[tap(id='tap_ref_0', source='ref_0', source_port='y', role='reference', visibility='primary', dimension=4)],
  metadata={'primary_output_node': 'out_0', 'primary_output_port': 'y'}
)
```

## Output shape

Each sample contains:

- `graph_dsl`, `graph_ast`, `graph_nodes`, `graph_edges`
- `io_ports`, `tap_specs`
- `module_params`
- `scenario`
- `t`
- `signals`
  - `external_inputs`
  - `primary_outputs`
  - `tap_signals`
  - `disturbance_signals`
  - `noise_signals`
- `teacher_signals`
- `system_view`
  - `node_models`
  - `assembled_linear_core`
  - `optional_hidden_states`
- `metrics`
  - `graph_metrics`
  - `system_metrics`
  - `tracking_metrics`
  - `channel_metrics`
- `tags`, `split_tags`

## Notes

- The generator is intentionally domain-agnostic; it models generic control graphs rather than a single industrial process.
- Feedback edges are delayed at graph-simulation time to avoid algebraic loops.
- Node dynamics are mostly linear state-space models; saturation and rate limits are handled locally during simulation.
- Hidden states are kept as optional teacher signals so the main task can remain “graph + scenario -> observed multi-tap time series”.
