# ControlGen

ControlGen is a mechanistic dataset generator for structure-to-time-series tasks in control systems.
The current version focuses on realistic SISO control chains built from a modular control graph:
reference, error summing junction, controller, actuator, plant, disturbance, sensor, and measurement noise.

## What is implemented

- Modular graph DSL with `loop`, `reference`, `sum`, `controller`, `actuator`, `plant`, `disturbance`, `sensor`, `noise`, and `tap`
- Low-level dynamic block support for `tf`, `gain`, `pid`, `delay`, `ss`, `series`, `parallel`, and `feedback`
- Parameterized module library for controller, actuator, plant, disturbance path, and sensor dynamics
- Realistic scenario sampling for reference signals, load disturbances, and measurement noise
- Multi-signal simulation producing `r`, `d`, `n`, `e`, `u_cmd`, `u_act`, `y`, and `y_m`
- Dataset schema with `graph_dsl`, `module_params`, `scenario`, `signals`, state-space module views, and mechanism-oriented metrics

## Quick start

```bash
python3 -m pip install -e .[dev]
python3 -m controlgen.cli --count 3 --seed 42
pytest
```

## Sample Graph DSL

```text
loop(
  reference=reference(kind='multistep', name='reference'),
  sum=sum(signs=[1, -1], name='error_sum'),
  controller=controller(kind='pid', name='controller'),
  actuator=actuator(kind='lag_saturation', name='actuator'),
  plant=plant(kind='delay_plus_lag', name='plant'),
  disturbance=disturbance(kind='load_step', injection='output', name='disturbance'),
  sensor=sensor(kind='lag', name='sensor'),
  noise=noise(kind='white_noise', name='measurement_noise'),
  taps=[tap(signal='r', name='tap_r'), tap(signal='y', name='tap_y')]
)
```

## Output shape

Each sample contains:

- `graph_dsl`: serialized modular control graph
- `graph_ast`: JSON-friendly graph tree
- `module_params`: instantiated controller, actuator, plant, disturbance path, and sensor parameters
- `scenario`: reference, disturbance, noise, and simulation settings
- `t`: time axis
- `signals`: `r`, `d`, `n`, `e`, `u_cmd`, `u_act`, `y`, `y_m`
- `system_view`: per-module linear state-space cores and nonlinear wrappers such as actuator saturation
- `metrics`: tracking, control effort, disturbance rejection, and measurement-chain metrics
- `tags`, `split_tags`: structure family, controller family, plant family, disturbance family, and reference family

## Notes

- The current release targets mechanistic SISO control loops.
- The linear core is simulated through discretized state-space blocks; actuator saturation is handled in the time-domain loop.
- Delays use a Padé approximation inside low-level dynamic modules.
