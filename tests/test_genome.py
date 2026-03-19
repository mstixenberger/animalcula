import random
from pathlib import Path

from animalcula import Config
from animalcula.sim.genome import (
    CreatureGenome,
    cluster_species,
    decode_genome,
    encode_creature_genome,
    genome_distance,
    genome_from_dict,
    genome_to_dict,
    mutate_genome,
)
from animalcula.sim.types import BrainState, CreatureState, EdgeState, NodeState, NodeType, Vec2


def test_encode_creature_genome_preserves_local_structure() -> None:
    nodes = [
        NodeState(
            position=Vec2(10.0, 10.0),
            velocity=Vec2.zero(),
            accumulated_force=Vec2.zero(),
            drag_coeff=1.0,
            radius=1.0,
            node_type=NodeType.MOUTH,
        ),
        NodeState(
            position=Vec2(16.0, 10.0),
            velocity=Vec2.zero(),
            accumulated_force=Vec2.zero(),
            drag_coeff=1.0,
            radius=1.5,
            node_type=NodeType.PHOTORECEPTOR,
        ),
    ]
    edges = [EdgeState(a=0, b=1, rest_length=6.0, stiffness=1.0, has_motor=True, motor_strength=2.0)]
    brain = BrainState(
        input_weights=((1.0, 2.0, 3.0, 4.0),),
        recurrent_weights=((0.5,),),
        biases=(0.1,),
        time_constants=(1.0,),
        states=(9.0,),
        output_size=1,
    )
    creature = CreatureState(node_indices=(0, 1), energy=5.0, brain=brain)

    genome = encode_creature_genome(nodes=nodes, edges=edges, creature=creature)

    assert isinstance(genome, CreatureGenome)
    assert len(genome.nodes) == 2
    assert genome.nodes[0].position == Vec2.zero()
    assert genome.nodes[1].position == Vec2(6.0, 0.0)
    assert genome.brain is not None
    assert genome.brain.states == (0.0,)
    assert genome.color_rgb == creature.color_rgb
    assert genome.visuals.silhouette_scale == 1.0
    assert genome.visuals.glyph_scale == 1.0
    assert genome.visuals.band_count == 2


def test_decode_genome_rebuilds_nodes_edges_and_brain() -> None:
    genome = CreatureGenome(
        nodes=(
            CreatureGenome.NodeGene(position=Vec2.zero(), radius=1.0, node_type=NodeType.BODY),
            CreatureGenome.NodeGene(position=Vec2(3.0, 4.0), radius=1.5, node_type=NodeType.MOUTH),
        ),
        edges=(
            CreatureGenome.EdgeGene(
                a=0,
                b=1,
                rest_length=5.0,
                stiffness=2.0,
                has_motor=True,
                motor_strength=1.5,
            ),
        ),
        brain=CreatureGenome.BrainGene(
            input_weights=((1.0, 0.0),),
            recurrent_weights=((0.5,),),
            biases=(0.2,),
            time_constants=(1.0,),
            output_size=1,
        ),
    )

    nodes, edges, brain = decode_genome(genome=genome, anchor_position=Vec2(10.0, 20.0), drag_coeff=1.0)

    assert nodes[0].position == Vec2(10.0, 20.0)
    assert nodes[1].position == Vec2(13.0, 24.0)
    assert edges[0].rest_length == 5.0
    assert brain is not None
    assert brain.states == (0.0,)


def test_mutate_genome_changes_shape_or_brain_but_keeps_bounds_valid() -> None:
    rng = random.Random(7)
    genome = CreatureGenome(
        nodes=(CreatureGenome.NodeGene(position=Vec2.zero(), radius=1.0, node_type=NodeType.BODY),),
        edges=(),
        brain=CreatureGenome.BrainGene(
            input_weights=((1.0, 2.0),),
            recurrent_weights=((0.5,),),
            biases=(0.1,),
            time_constants=(1.0,),
            output_size=1,
        ),
    )

    mutated = mutate_genome(
        genome=genome,
        rng=rng,
        position_sigma=0.5,
        radius_sigma=0.05,
        weight_sigma=0.1,
        bias_sigma=0.05,
        tau_sigma=0.02,
        motor_strength_sigma=0.2,
    )

    assert mutated != genome
    assert mutated.nodes[0].radius > 0.0
    assert mutated.brain is not None
    assert all(value > 0.0 for value in mutated.brain.time_constants)
    assert all(0 <= channel <= 255 for channel in mutated.color_rgb)
    assert 0.85 <= mutated.visuals.silhouette_scale <= 1.5
    assert 0.85 <= mutated.visuals.glyph_scale <= 1.75
    assert 1 <= mutated.visuals.band_count <= 4
    assert 0.0 <= mutated.visuals.band_offset < 1.0


def test_genome_dict_roundtrip_preserves_lineage_color() -> None:
    genome = CreatureGenome(
        nodes=(CreatureGenome.NodeGene(position=Vec2.zero(), radius=1.0, node_type=NodeType.BODY),),
        edges=(),
        brain=None,
        color_rgb=(45, 120, 200),
    )

    restored = genome_from_dict(genome_to_dict(genome))

    assert restored == genome


def test_genome_dict_roundtrip_preserves_visual_traits() -> None:
    genome = CreatureGenome(
        nodes=(CreatureGenome.NodeGene(position=Vec2.zero(), radius=1.0, node_type=NodeType.BODY),),
        edges=(),
        brain=None,
        visuals=CreatureGenome.VisualGene(
            silhouette_scale=1.25,
            glyph_scale=1.4,
            band_count=4,
            band_offset=0.35,
        ),
    )

    restored = genome_from_dict(genome_to_dict(genome))

    assert restored == genome


def test_mutate_genome_can_add_a_structural_body_node() -> None:
    rng = random.Random(7)
    genome = CreatureGenome(
        nodes=(CreatureGenome.NodeGene(position=Vec2.zero(), radius=1.0, node_type=NodeType.BODY),),
        edges=(),
        brain=None,
    )

    mutated = mutate_genome(
        genome=genome,
        rng=rng,
        position_sigma=0.0,
        radius_sigma=0.0,
        weight_sigma=0.0,
        bias_sigma=0.0,
        tau_sigma=0.0,
        motor_strength_sigma=0.0,
        structural_mutation_rate=1.0,
    )

    assert len(mutated.nodes) == 2
    assert len(mutated.edges) == 1


def test_mutate_genome_can_change_a_node_type() -> None:
    rng = random.Random(7)
    genome = CreatureGenome(
        nodes=(CreatureGenome.NodeGene(position=Vec2.zero(), radius=1.0, node_type=NodeType.BODY),),
        edges=(),
        brain=None,
    )

    mutated = mutate_genome(
        genome=genome,
        rng=rng,
        position_sigma=0.0,
        radius_sigma=0.0,
        weight_sigma=0.0,
        bias_sigma=0.0,
        tau_sigma=0.0,
        motor_strength_sigma=0.0,
        node_type_mutation_rate=1.0,
    )

    assert mutated.nodes[0].node_type != NodeType.BODY


def test_mutate_genome_preserves_extra_control_channels_when_resizing_outputs() -> None:
    rng = random.Random(7)
    genome = CreatureGenome(
        nodes=(
            CreatureGenome.NodeGene(position=Vec2.zero(), radius=1.0, node_type=NodeType.GRIPPER),
            CreatureGenome.NodeGene(position=Vec2(2.0, 0.0), radius=1.0, node_type=NodeType.MOUTH),
        ),
        edges=(
            CreatureGenome.EdgeGene(
                a=0,
                b=1,
                rest_length=2.0,
                stiffness=1.0,
                has_motor=True,
                motor_strength=1.0,
            ),
        ),
        brain=CreatureGenome.BrainGene(
            input_weights=((0.0,) * 16,),
            recurrent_weights=((0.0,),),
            biases=(0.0,),
            time_constants=(1.0,),
            output_size=5,
        ),
    )

    mutated = mutate_genome(
        genome=genome,
        rng=rng,
        position_sigma=0.0,
        radius_sigma=0.0,
        weight_sigma=0.0,
        bias_sigma=0.0,
        tau_sigma=0.0,
        motor_strength_sigma=0.0,
        node_type_mutation_rate=0.0,
    )

    assert mutated.brain is not None
    assert mutated.brain.output_size == 5


def test_mutate_genome_expands_outputs_to_match_morphology() -> None:
    rng = random.Random(7)
    genome = CreatureGenome(
        nodes=(
            CreatureGenome.NodeGene(position=Vec2.zero(), radius=1.0, node_type=NodeType.GRIPPER),
            CreatureGenome.NodeGene(position=Vec2(2.0, 0.0), radius=1.0, node_type=NodeType.MOUTH),
        ),
        edges=(
            CreatureGenome.EdgeGene(
                a=0,
                b=1,
                rest_length=2.0,
                stiffness=1.0,
                has_motor=True,
                motor_strength=1.0,
            ),
        ),
        brain=CreatureGenome.BrainGene(
            input_weights=((0.0,) * 16,),
            recurrent_weights=((0.0,),),
            biases=(0.0,),
            time_constants=(1.0,),
            output_size=1,
        ),
    )

    mutated = mutate_genome(
        genome=genome,
        rng=rng,
        position_sigma=0.0,
        radius_sigma=0.0,
        weight_sigma=0.0,
        bias_sigma=0.0,
        tau_sigma=0.0,
        motor_strength_sigma=0.0,
        node_type_mutation_rate=0.0,
    )

    assert mutated.brain is not None
    assert mutated.brain.output_size == 3


def test_mutate_genome_can_toggle_a_motorized_edge() -> None:
    rng = random.Random(7)
    genome = CreatureGenome(
        nodes=(
            CreatureGenome.NodeGene(position=Vec2.zero(), radius=1.0, node_type=NodeType.BODY),
            CreatureGenome.NodeGene(position=Vec2(2.0, 0.0), radius=1.0, node_type=NodeType.BODY),
        ),
        edges=(
            CreatureGenome.EdgeGene(
                a=0,
                b=1,
                rest_length=2.0,
                stiffness=1.0,
                has_motor=False,
                motor_strength=0.0,
            ),
        ),
        brain=CreatureGenome.BrainGene(
            input_weights=((0.0,) * 16,),
            recurrent_weights=((0.0,),),
            biases=(0.0,),
            time_constants=(1.0,),
            output_size=0,
        ),
    )

    mutated = mutate_genome(
        genome=genome,
        rng=rng,
        position_sigma=0.0,
        radius_sigma=0.0,
        weight_sigma=0.0,
        bias_sigma=0.0,
        tau_sigma=0.0,
        motor_strength_sigma=0.0,
        node_type_mutation_rate=0.0,
        motor_toggle_mutation_rate=1.0,
    )

    assert mutated.edges[0].has_motor is True
    assert mutated.brain is not None
    assert mutated.brain.output_size == 1


def test_mutate_genome_can_grow_hidden_neuron_prefix_without_changing_outputs() -> None:
    rng = random.Random(7)
    genome = CreatureGenome(
        nodes=(CreatureGenome.NodeGene(position=Vec2.zero(), radius=1.0, node_type=NodeType.BODY),),
        edges=(),
        brain=CreatureGenome.BrainGene(
            input_weights=((1.0, 2.0),),
            recurrent_weights=((0.5,),),
            biases=(0.1,),
            time_constants=(1.0,),
            output_size=1,
            states=(0.0,),
        ),
    )

    mutated = mutate_genome(
        genome=genome,
        rng=rng,
        position_sigma=0.0,
        radius_sigma=0.0,
        weight_sigma=0.0,
        bias_sigma=0.0,
        tau_sigma=0.0,
        motor_strength_sigma=0.0,
        hidden_neuron_mutation_rate=1.0,
        max_hidden_neurons=24,
    )

    assert mutated.brain is not None
    assert mutated.brain.output_size == 1
    assert len(mutated.brain.biases) == 2
    assert mutated.brain.input_weights[-1] == genome.brain.input_weights[-1]
    assert mutated.brain.recurrent_weights[-1][-1] == genome.brain.recurrent_weights[-1][-1]


def test_mutate_genome_can_shrink_hidden_neuron_prefix_at_capacity() -> None:
    rng = random.Random(7)
    genome = CreatureGenome(
        nodes=(CreatureGenome.NodeGene(position=Vec2.zero(), radius=1.0, node_type=NodeType.BODY),),
        edges=(),
        brain=CreatureGenome.BrainGene(
            input_weights=(
                (10.0, 10.0),
                (20.0, 20.0),
                (30.0, 30.0),
            ),
            recurrent_weights=(
                (1.0, 2.0, 3.0),
                (4.0, 5.0, 6.0),
                (7.0, 8.0, 9.0),
            ),
            biases=(0.1, 0.2, 0.3),
            time_constants=(1.0, 2.0, 3.0),
            output_size=1,
            states=(0.0, 1.0, 2.0),
        ),
    )

    mutated = mutate_genome(
        genome=genome,
        rng=rng,
        position_sigma=0.0,
        radius_sigma=0.0,
        weight_sigma=0.0,
        bias_sigma=0.0,
        tau_sigma=0.0,
        motor_strength_sigma=0.0,
        hidden_neuron_mutation_rate=1.0,
        max_hidden_neurons=2,
    )

    assert mutated.brain is not None
    assert mutated.brain.output_size == 1
    assert len(mutated.brain.biases) == 2
    assert mutated.brain.input_weights[0] == (20.0, 20.0)
    assert mutated.brain.input_weights[-1] == (30.0, 30.0)
    assert mutated.brain.recurrent_weights[-1] == (8.0, 9.0)


def test_default_config_mutation_rates_allow_topology_or_role_exploration() -> None:
    config = Config.from_yaml(Path("config/default.yaml"))
    genome = CreatureGenome(
        nodes=(
            CreatureGenome.NodeGene(position=Vec2.zero(), radius=1.0, node_type=NodeType.BODY),
            CreatureGenome.NodeGene(position=Vec2(2.0, 0.0), radius=1.0, node_type=NodeType.BODY),
        ),
        edges=(
            CreatureGenome.EdgeGene(
                a=0,
                b=1,
                rest_length=2.0,
                stiffness=1.0,
                has_motor=False,
                motor_strength=0.0,
            ),
        ),
        brain=CreatureGenome.BrainGene(
            input_weights=((0.0,) * 16,),
            recurrent_weights=((0.0,),),
            biases=(0.0,),
            time_constants=(1.0,),
            output_size=0,
        ),
    )
    rng = random.Random(7)

    saw_exploration = False
    for _ in range(64):
        mutated = mutate_genome(
            genome=genome,
            rng=rng,
            position_sigma=config.evolution.position_mutation_sigma,
            radius_sigma=config.evolution.radius_mutation_sigma,
            weight_sigma=0.0,
            bias_sigma=0.0,
            tau_sigma=0.0,
            motor_strength_sigma=0.0,
            motor_toggle_mutation_rate=config.evolution.motor_toggle_mutation_rate,
            node_type_mutation_rate=config.evolution.node_type_mutation_rate,
            structural_mutation_rate=config.evolution.structural_mutation_rate,
        )
        if (
            len(mutated.nodes) != len(genome.nodes)
            or any(
                mutated_node.node_type != original_node.node_type
                for mutated_node, original_node in zip(mutated.nodes, genome.nodes, strict=False)
            )
            or any(
                mutated_edge.has_motor != original_edge.has_motor
                for mutated_edge, original_edge in zip(mutated.edges, genome.edges, strict=True)
            )
        ):
            saw_exploration = True
            break

    assert saw_exploration is True


def test_genome_distance_is_zero_for_identical_genomes() -> None:
    genome = CreatureGenome(
        nodes=(CreatureGenome.NodeGene(position=Vec2.zero(), radius=1.0, node_type=NodeType.BODY),),
        edges=(),
        brain=None,
    )

    assert genome_distance(genome, genome) == 0.0


def test_cluster_species_groups_similar_genomes() -> None:
    genome_a = CreatureGenome(
        nodes=(CreatureGenome.NodeGene(position=Vec2.zero(), radius=1.0, node_type=NodeType.BODY),),
        edges=(),
        brain=None,
    )
    genome_b = CreatureGenome(
        nodes=(CreatureGenome.NodeGene(position=Vec2.zero(), radius=1.05, node_type=NodeType.BODY),),
        edges=(),
        brain=None,
    )
    genome_c = CreatureGenome(
        nodes=(
            CreatureGenome.NodeGene(position=Vec2.zero(), radius=1.0, node_type=NodeType.MOUTH),
            CreatureGenome.NodeGene(position=Vec2(2.0, 0.0), radius=1.0, node_type=NodeType.GRIPPER),
        ),
        edges=(CreatureGenome.EdgeGene(a=0, b=1, rest_length=2.0, stiffness=1.0),),
        brain=None,
    )

    labels = cluster_species((genome_a, genome_b, genome_c), threshold=0.5)

    assert labels[0] == labels[1]
    assert labels[0] != labels[2]


def _make_chain_genome(n: int, *, node_type: NodeType = NodeType.MOUTH) -> CreatureGenome:
    """Build a linear chain of n nodes connected end-to-end."""
    nodes = tuple(
        CreatureGenome.NodeGene(
            position=Vec2(float(i) * 3.0, 0.0),
            radius=1.0,
            node_type=node_type if i == n - 1 else NodeType.BODY,
        )
        for i in range(n)
    )
    edges = tuple(
        CreatureGenome.EdgeGene(
            a=i, b=i + 1, rest_length=3.0, stiffness=1.0, has_motor=True, motor_strength=1.0
        )
        for i in range(n - 1)
    )
    brain = CreatureGenome.BrainGene(
        input_weights=tuple((0.0,) * 16 for _ in range(n - 1)),
        recurrent_weights=tuple(tuple(0.0 for _ in range(n - 1)) for _ in range(n - 1)),
        biases=tuple(0.0 for _ in range(n - 1)),
        time_constants=tuple(1.0 for _ in range(n - 1)),
        output_size=n - 1,
    )
    return CreatureGenome(nodes=nodes, edges=edges, brain=brain)


def test_chain_extension_extends_terminal_node() -> None:
    rng = random.Random(42)
    genome = _make_chain_genome(3)

    mutated = mutate_genome(
        genome=genome,
        rng=rng,
        position_sigma=0.0,
        radius_sigma=0.0,
        weight_sigma=0.0,
        bias_sigma=0.0,
        tau_sigma=0.0,
        motor_strength_sigma=0.0,
        chain_extension_mutation_rate=1.0,
    )

    assert len(mutated.nodes) == 4
    assert len(mutated.edges) == 3
    new_edge = mutated.edges[-1]
    assert new_edge.has_motor is True
    assert new_edge.motor_strength >= 0.5


def test_chain_extension_inherits_node_type() -> None:
    rng = random.Random(42)
    genome = _make_chain_genome(3, node_type=NodeType.GRIPPER)

    mutated = mutate_genome(
        genome=genome,
        rng=rng,
        position_sigma=0.0,
        radius_sigma=0.0,
        weight_sigma=0.0,
        bias_sigma=0.0,
        tau_sigma=0.0,
        motor_strength_sigma=0.0,
        chain_extension_mutation_rate=1.0,
    )

    terminal_types = [
        mutated.nodes[i].node_type
        for i in range(len(mutated.nodes))
        if i == len(mutated.nodes) - 1 or i == len(mutated.nodes) - 2
    ]
    assert NodeType.GRIPPER in terminal_types


def test_chain_extension_respects_max_nodes() -> None:
    rng = random.Random(42)
    genome = _make_chain_genome(4)

    mutated = mutate_genome(
        genome=genome,
        rng=rng,
        position_sigma=0.0,
        radius_sigma=0.0,
        weight_sigma=0.0,
        bias_sigma=0.0,
        tau_sigma=0.0,
        motor_strength_sigma=0.0,
        chain_extension_mutation_rate=1.0,
        max_nodes_per_creature=4,
    )

    assert len(mutated.nodes) == 4


def test_chain_extension_skips_without_terminals() -> None:
    """A fully triangulated graph has no terminal (degree-1) nodes."""
    rng = random.Random(42)
    genome = CreatureGenome(
        nodes=(
            CreatureGenome.NodeGene(position=Vec2(0.0, 0.0), radius=1.0, node_type=NodeType.BODY),
            CreatureGenome.NodeGene(position=Vec2(3.0, 0.0), radius=1.0, node_type=NodeType.BODY),
            CreatureGenome.NodeGene(position=Vec2(1.5, 2.6), radius=1.0, node_type=NodeType.BODY),
        ),
        edges=(
            CreatureGenome.EdgeGene(a=0, b=1, rest_length=3.0, stiffness=1.0),
            CreatureGenome.EdgeGene(a=1, b=2, rest_length=3.0, stiffness=1.0),
            CreatureGenome.EdgeGene(a=2, b=0, rest_length=3.0, stiffness=1.0),
        ),
        brain=None,
    )

    mutated = mutate_genome(
        genome=genome,
        rng=rng,
        position_sigma=0.0,
        radius_sigma=0.0,
        weight_sigma=0.0,
        bias_sigma=0.0,
        tau_sigma=0.0,
        motor_strength_sigma=0.0,
        chain_extension_mutation_rate=1.0,
    )

    assert len(mutated.nodes) == 3


def test_structural_mutation_respects_max_nodes() -> None:
    rng = random.Random(42)
    genome = _make_chain_genome(4)

    mutated = mutate_genome(
        genome=genome,
        rng=rng,
        position_sigma=0.0,
        radius_sigma=0.0,
        weight_sigma=0.0,
        bias_sigma=0.0,
        tau_sigma=0.0,
        motor_strength_sigma=0.0,
        structural_mutation_rate=1.0,
        max_nodes_per_creature=4,
    )

    assert len(mutated.nodes) == 4


def test_chain_extension_warm_starts_recurrent_coupling() -> None:
    rng = random.Random(42)
    genome = _make_chain_genome(3)

    mutated = mutate_genome(
        genome=genome,
        rng=rng,
        position_sigma=0.0,
        radius_sigma=0.0,
        weight_sigma=0.0,
        bias_sigma=0.0,
        tau_sigma=0.0,
        motor_strength_sigma=0.0,
        chain_extension_mutation_rate=1.0,
    )

    assert mutated.brain is not None
    # At least one neuron should have nonzero recurrent coupling from another
    # (the warm-start sets recurrent_weights[new_motor][parent_motor])
    found_coupling = False
    for row in mutated.brain.recurrent_weights:
        if any(w != 0.0 for w in row):
            found_coupling = True
            break
    assert found_coupling, "Expected nonzero recurrent coupling from warm-start"


def test_chain_extension_offsets_time_constant() -> None:
    rng = random.Random(42)
    genome = _make_chain_genome(3)

    mutated = mutate_genome(
        genome=genome,
        rng=rng,
        position_sigma=0.0,
        radius_sigma=0.0,
        weight_sigma=0.0,
        bias_sigma=0.0,
        tau_sigma=0.0,
        motor_strength_sigma=0.0,
        chain_extension_mutation_rate=1.0,
    )

    assert mutated.brain is not None
    # The new motor's time constant should differ from 1.0 (the default)
    taus = mutated.brain.time_constants
    assert any(tau != 1.0 for tau in taus), "Expected at least one offset time constant from warm-start"


def test_genome_node_gene_has_drag_coeff_with_default() -> None:
    """GenomeNodeGene should support an optional drag_coeff field."""
    node = CreatureGenome.NodeGene(position=Vec2.zero(), radius=1.0, node_type=NodeType.BODY)
    assert node.drag_coeff == 1.0  # default

    node_custom = CreatureGenome.NodeGene(
        position=Vec2.zero(), radius=1.0, node_type=NodeType.BODY, drag_coeff=2.5
    )
    assert node_custom.drag_coeff == 2.5


def test_decode_genome_uses_per_node_drag() -> None:
    """decode_genome should use the node gene's drag_coeff, not the config default."""
    genome = CreatureGenome(
        nodes=(
            CreatureGenome.NodeGene(position=Vec2.zero(), radius=1.0, node_type=NodeType.BODY, drag_coeff=2.0),
            CreatureGenome.NodeGene(position=Vec2(3.0, 0.0), radius=1.0, node_type=NodeType.BODY, drag_coeff=4.0),
        ),
        edges=(),
        brain=None,
    )

    nodes, _, _ = decode_genome(genome=genome, anchor_position=Vec2.zero(), drag_coeff=1.0)

    assert nodes[0].drag_coeff == 2.0
    assert nodes[1].drag_coeff == 4.0


def test_encode_creature_genome_preserves_drag_coeff() -> None:
    """encode_creature_genome should capture drag_coeff from runtime nodes."""
    nodes = [
        NodeState(
            position=Vec2(10.0, 10.0), velocity=Vec2.zero(),
            accumulated_force=Vec2.zero(), drag_coeff=3.5, radius=1.0,
        ),
    ]
    creature = CreatureState(node_indices=(0,), energy=1.0)

    genome = encode_creature_genome(nodes=nodes, edges=[], creature=creature)
    assert genome.nodes[0].drag_coeff == 3.5


def test_mutate_genome_perturbs_drag_within_bounds() -> None:
    """Drag mutation should perturb drag and clamp to [0.5, 5.0]."""
    rng = random.Random(42)
    genome = CreatureGenome(
        nodes=(
            CreatureGenome.NodeGene(position=Vec2.zero(), radius=1.0, node_type=NodeType.BODY, drag_coeff=1.0),
        ),
        edges=(),
        brain=None,
    )

    saw_change = False
    for _ in range(50):
        mutated = mutate_genome(
            genome=genome, rng=rng,
            position_sigma=0.0, radius_sigma=0.0, weight_sigma=0.0,
            bias_sigma=0.0, tau_sigma=0.0, motor_strength_sigma=0.0,
            drag_mutation_sigma=0.3,
        )
        if mutated.nodes[0].drag_coeff != genome.nodes[0].drag_coeff:
            saw_change = True
        assert 0.5 <= mutated.nodes[0].drag_coeff <= 5.0

    assert saw_change, "Expected drag to be perturbed at least once in 50 tries"


def test_genome_dict_roundtrip_preserves_drag_coeff() -> None:
    genome = CreatureGenome(
        nodes=(
            CreatureGenome.NodeGene(
                position=Vec2.zero(), radius=1.0, node_type=NodeType.BODY, drag_coeff=2.5
            ),
        ),
        edges=(),
        brain=None,
    )

    restored = genome_from_dict(genome_to_dict(genome))
    assert restored.nodes[0].drag_coeff == 2.5


def test_genome_from_dict_backward_compat_missing_drag() -> None:
    """Old genome dicts without drag_coeff should default to 1.0."""
    payload = {
        "nodes": [{"position": [0.0, 0.0], "radius": 1.0, "node_type": "body"}],
        "edges": [],
        "brain": None,
    }
    genome = genome_from_dict(payload)
    assert genome.nodes[0].drag_coeff == 1.0


def test_genome_distance_includes_drag_difference() -> None:
    """genome_distance should account for mean drag difference."""
    genome_a = CreatureGenome(
        nodes=(CreatureGenome.NodeGene(position=Vec2.zero(), radius=1.0, node_type=NodeType.BODY, drag_coeff=1.0),),
        edges=(), brain=None,
    )
    genome_b = CreatureGenome(
        nodes=(CreatureGenome.NodeGene(position=Vec2.zero(), radius=1.0, node_type=NodeType.BODY, drag_coeff=3.0),),
        edges=(), brain=None,
    )

    dist = genome_distance(genome_a, genome_b)
    dist_same = genome_distance(genome_a, genome_a)
    assert dist > dist_same


def test_creature_genome_has_mutation_rate_with_default() -> None:
    """CreatureGenome should have a mutation_rate field defaulting to 0.05."""
    genome = CreatureGenome(
        nodes=(CreatureGenome.NodeGene(position=Vec2.zero(), radius=1.0, node_type=NodeType.BODY),),
        edges=(), brain=None,
    )
    assert genome.mutation_rate == 0.05


def test_mutation_rate_meta_gene_mutates_each_generation() -> None:
    """The mutation_rate should self-adapt via log-normal perturbation."""
    rng = random.Random(42)
    genome = CreatureGenome(
        nodes=(CreatureGenome.NodeGene(position=Vec2.zero(), radius=1.0, node_type=NodeType.BODY),),
        edges=(), brain=None,
        mutation_rate=0.05,
    )

    saw_change = False
    for _ in range(20):
        mutated = mutate_genome(
            genome=genome, rng=rng,
            position_sigma=0.0, radius_sigma=0.0, weight_sigma=0.1,
            bias_sigma=0.05, tau_sigma=0.02, motor_strength_sigma=0.2,
        )
        if mutated.mutation_rate != genome.mutation_rate:
            saw_change = True
            break

    assert saw_change, "Expected mutation_rate to self-adapt"


def test_mutation_rate_stays_in_bounds() -> None:
    """mutation_rate should be clamped to [0.001, 0.2]."""
    rng = random.Random(42)
    genome = CreatureGenome(
        nodes=(CreatureGenome.NodeGene(position=Vec2.zero(), radius=1.0, node_type=NodeType.BODY),),
        edges=(), brain=None,
        mutation_rate=0.05,
    )

    for _ in range(200):
        genome = mutate_genome(
            genome=genome, rng=rng,
            position_sigma=0.0, radius_sigma=0.0, weight_sigma=0.1,
            bias_sigma=0.05, tau_sigma=0.02, motor_strength_sigma=0.2,
        )
        assert 0.001 <= genome.mutation_rate <= 0.2


def test_mutation_rate_scales_parametric_sigmas() -> None:
    """Parametric mutations (weight, bias, tau, drag) should scale with mutation_rate."""
    rng_low = random.Random(99)
    rng_high = random.Random(99)

    brain = CreatureGenome.BrainGene(
        input_weights=((0.0,) * 16,),
        recurrent_weights=((0.0,),),
        biases=(0.0,),
        time_constants=(1.0,),
        output_size=1,
    )
    genome_low = CreatureGenome(
        nodes=(CreatureGenome.NodeGene(position=Vec2.zero(), radius=1.0, node_type=NodeType.BODY),),
        edges=(), brain=brain, mutation_rate=0.01,
    )
    genome_high = CreatureGenome(
        nodes=(CreatureGenome.NodeGene(position=Vec2.zero(), radius=1.0, node_type=NodeType.BODY),),
        edges=(), brain=brain, mutation_rate=0.2,
    )

    mutated_low = mutate_genome(
        genome=genome_low, rng=rng_low,
        position_sigma=0.0, radius_sigma=0.0, weight_sigma=0.1,
        bias_sigma=0.05, tau_sigma=0.02, motor_strength_sigma=0.2,
    )
    mutated_high = mutate_genome(
        genome=genome_high, rng=rng_high,
        position_sigma=0.0, radius_sigma=0.0, weight_sigma=0.1,
        bias_sigma=0.05, tau_sigma=0.02, motor_strength_sigma=0.2,
    )

    # Higher mutation_rate should produce larger weight deviations on average
    low_delta = abs(mutated_low.brain.biases[0] - brain.biases[0])
    high_delta = abs(mutated_high.brain.biases[0] - brain.biases[0])
    assert high_delta > low_delta


def test_structural_rates_unaffected_by_mutation_rate() -> None:
    """Structural mutation rates stay config-driven, not scaled by mutation_rate."""
    rng_a = random.Random(42)
    rng_b = random.Random(42)

    genome = CreatureGenome(
        nodes=(
            CreatureGenome.NodeGene(position=Vec2.zero(), radius=1.0, node_type=NodeType.BODY),
            CreatureGenome.NodeGene(position=Vec2(3.0, 0.0), radius=1.0, node_type=NodeType.BODY),
        ),
        edges=(CreatureGenome.EdgeGene(a=0, b=1, rest_length=3.0, stiffness=1.0),),
        brain=None,
    )

    genome_low = CreatureGenome(
        nodes=genome.nodes, edges=genome.edges, brain=genome.brain,
        mutation_rate=0.001,
    )
    genome_high = CreatureGenome(
        nodes=genome.nodes, edges=genome.edges, brain=genome.brain,
        mutation_rate=0.2,
    )

    # With identical seeds and structural_mutation_rate=1.0, both should add a node
    mutated_low = mutate_genome(
        genome=genome_low, rng=rng_a,
        position_sigma=0.0, radius_sigma=0.0, weight_sigma=0.0,
        bias_sigma=0.0, tau_sigma=0.0, motor_strength_sigma=0.0,
        structural_mutation_rate=1.0,
    )
    mutated_high = mutate_genome(
        genome=genome_high, rng=rng_b,
        position_sigma=0.0, radius_sigma=0.0, weight_sigma=0.0,
        bias_sigma=0.0, tau_sigma=0.0, motor_strength_sigma=0.0,
        structural_mutation_rate=1.0,
    )

    assert len(mutated_low.nodes) == 3
    assert len(mutated_high.nodes) == 3


def test_genome_dict_roundtrip_preserves_mutation_rate() -> None:
    genome = CreatureGenome(
        nodes=(CreatureGenome.NodeGene(position=Vec2.zero(), radius=1.0, node_type=NodeType.BODY),),
        edges=(), brain=None, mutation_rate=0.08,
    )

    restored = genome_from_dict(genome_to_dict(genome))
    assert restored.mutation_rate == 0.08


def test_genome_from_dict_backward_compat_missing_mutation_rate() -> None:
    """Old genome dicts without mutation_rate should default to 0.05."""
    payload = {
        "nodes": [{"position": [0.0, 0.0], "radius": 1.0, "node_type": "body"}],
        "edges": [], "brain": None,
    }
    genome = genome_from_dict(payload)
    assert genome.mutation_rate == 0.05
