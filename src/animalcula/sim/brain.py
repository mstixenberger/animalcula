"""CTRNN runtime helpers."""

from __future__ import annotations

from dataclasses import replace
import math

from animalcula.sim.types import BrainState


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def step_brain(brain: BrainState, inputs: tuple[float, ...], dt: float) -> tuple[BrainState, tuple[float, ...]]:
    input_size = len(brain.input_weights[0]) if brain.input_weights else 0
    if len(inputs) < input_size:
        inputs = inputs + ((0.0,) * (input_size - len(inputs)))
    elif len(inputs) > input_size:
        inputs = inputs[:input_size]

    activations = tuple(sigmoid(state + bias) for state, bias in zip(brain.states, brain.biases, strict=True))
    new_states: list[float] = []

    for index, state in enumerate(brain.states):
        recurrent_term = sum(
            weight * activation
            for weight, activation in zip(brain.recurrent_weights[index], activations, strict=True)
        )
        input_term = sum(
            weight * value
            for weight, value in zip(brain.input_weights[index], inputs, strict=True)
        )
        delta = (dt / brain.time_constants[index]) * (-state + recurrent_term + input_term)
        new_states.append(state + delta)

    outputs = tuple(sigmoid(state + bias) for state, bias in zip(new_states, brain.biases, strict=True))
    updated = replace(brain, states=tuple(new_states))
    return updated, outputs[-brain.output_size :]
