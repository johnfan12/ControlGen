"""ControlGen public API."""

from controlgen.dataset import DatasetSample, build_sample
from controlgen.dsl import ast_to_dict, parse_dsl, serialize_dsl
from controlgen.generator import (
    DatasetConfig,
    GrammarConfig,
    SamplingConfig,
    generate_graph,
    generate_dataset,
    generate_sample,
    generate_structure,
    sample_parameters,
    sample_scenario,
)
from controlgen.simulate import ScenarioSpec, SignalBundle, SignalSpec, SimulationConfig, simulate
from controlgen.transfer_function import (
    ParameterizedSystem,
    RationalTransferFunction,
    StateSpaceModel,
    tf_from_node,
)
from controlgen.types import ControlGraph, ControlNode, ParameterizedGraph

__all__ = [
    "ControlGraph",
    "ControlNode",
    "DatasetConfig",
    "DatasetSample",
    "GrammarConfig",
    "ParameterizedSystem",
    "ParameterizedGraph",
    "RationalTransferFunction",
    "SamplingConfig",
    "ScenarioSpec",
    "SignalBundle",
    "SignalSpec",
    "SimulationConfig",
    "StateSpaceModel",
    "ast_to_dict",
    "build_sample",
    "generate_graph",
    "generate_dataset",
    "generate_sample",
    "generate_structure",
    "parse_dsl",
    "sample_parameters",
    "sample_scenario",
    "serialize_dsl",
    "simulate",
    "tf_from_node",
]
