import math

from animalcula.sim.brain import step_brain
from animalcula.sim.types import BrainState


def test_ctrnn_step_updates_state_and_output() -> None:
    brain = BrainState(
        input_weights=((1.0,),),
        recurrent_weights=((0.0,),),
        biases=(0.0,),
        time_constants=(1.0,),
        states=(0.0,),
        output_size=1,
    )

    updated, outputs = step_brain(brain=brain, inputs=(1.0,), dt=0.5)

    assert math.isclose(updated.states[0], 0.5, rel_tol=1e-6)
    assert math.isclose(outputs[0], 1.0 / (1.0 + math.exp(-0.5)), rel_tol=1e-6)


def test_ctrnn_respects_recurrent_feedback() -> None:
    brain = BrainState(
        input_weights=((0.0,),),
        recurrent_weights=((2.0,),),
        biases=(0.0,),
        time_constants=(2.0,),
        states=(1.0,),
        output_size=1,
    )

    updated, outputs = step_brain(brain=brain, inputs=(0.0,), dt=1.0)

    expected_activation = 1.0 / (1.0 + math.exp(-1.0))
    expected_state = 1.0 + (1.0 / 2.0) * (-1.0 + (2.0 * expected_activation))

    assert math.isclose(updated.states[0], expected_state, rel_tol=1e-6)
    assert math.isclose(outputs[0], 1.0 / (1.0 + math.exp(-expected_state)), rel_tol=1e-6)
