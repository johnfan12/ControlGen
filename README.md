# ControlGen

ControlGen is a lightweight V1 dataset generator for control-structure-to-time-series tasks.
It generates linear SISO control systems from a symbolic `ControlDSL`, samples stable
parameters, simulates standard test inputs, and exports paired structure/trajectory samples.

## What is implemented

- Function-style `ControlDSL` for `tf`, `gain`, `pid`, `delay`, `series`, `parallel`, `feedback`
- Stable parameter sampling for plant/controller templates
- Continuous-time transfer-function algebra for composition and closed-loop reduction
- Standard input simulation for `step`, `impulse`, `ramp`, `sine`
- Dataset sample schema with trajectories, poles/zeros, and step-response metrics
- CLI for exporting JSONL samples

## Quick start

```bash
python3 -m pip install -e .[dev]
python3 -m controlgen.cli --count 3 --seed 42
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
- `input_spec`: input family and parameters
- `t`, `u`, `y`: time axis, input trajectory, output trajectory
- `metrics`: stability, poles, zeros, final value, and step metrics when applicable
- `tags`: closed-loop flag, controller type, order, and difficulty

## Notes

- V1 currently targets continuous-time linear SISO systems.
- Delays use a Padé approximation when enabled in the structure generator.
- The default generator emphasizes stable closed-loop samples suitable for sequence modeling.
