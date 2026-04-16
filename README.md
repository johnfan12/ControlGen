# ControlGen

ControlGen is a dataset generator for control-structure-to-time-series tasks. The current
version uses a state-space simulation core, supports richer linear SISO structure families,
and can also generate minimal 2x2 linear MIMO closed-loop systems.

## What is implemented

- Function-style `ControlDSL` for `tf`, `gain`, `matgain`, `pid`, `delay`, `ss`, `series`, `parallel`, `feedback`
- Unified continuous-time state-space backend for SISO and 2x2 MIMO systems
- Richer SISO structure families: PID feedback, lead-lag, cascade compensator, feedforward, two-degree-of-freedom, sensor filtering
- Parameter families for `balanced`, `fast`, `oscillatory`, and `stiff` dynamics
- Input families for `standard` and `benchmark` trajectories
- Dataset samples with 2D `u/y`, state-space views, split tags, and nested metrics
- CLI and notebook support for both SISO and MIMO visualization

## Quick start

```bash
python3 -m pip install -e .[dev]
python3 -m controlgen.cli --count 3 --seed 42 --system-mode mixed
pytest
```

## Sample DSL

```text
feedback(
  forward=series(
    pid(kp=1.0, ki=0.3, kd=0.1, tau=0.05, name='controller'),
    tf(num=[1.0], den=[1.0, 1.5, 0.5], name='plant')
  ),
  feedback=gain(k=1.0, name='sensor'),
  sign=-1
)
```

## Output shape

Each sample contains:

- `dsl_text`: serialized control structure
- `ast`: JSON-friendly tree form
- `system_view`: DSL, AST, and state-space matrices
- `input_spec`: input family and parameters
- `input_channels`, `output_channels`: I/O dimensions
- `t`, `u`, `y`: time axis and 2D input/output trajectories
- `metrics`: nested system metrics and per-output channel metrics
- `tags`, `split_tags`: structure family, parameter family, input family, controller family, and I/O shape

## Notes

- The backend targets continuous-time linear systems.
- MIMO support is currently limited to 2x2 coupled systems with diagonal PID or static decoupler plus diagonal PID control.
- Delays use a Padé approximation when enabled in the structure generator.
