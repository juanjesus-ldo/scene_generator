import bpy
import random
import numpy as np

from mathutils import Euler
from math import radians

def init_lights(lightning_position_constraints):
    """
    Initialize lights in the scene based on provided lightning position constraints.

    This function creates a World if one does not exist and sets up its background node.
    It then processes the given lightning constraints (groups of 6 elements: light type, x, y, z, min intensity, max intensity)
    to add lights to the scene with a random energy value within the specified range.

    Args:
        lightning_position_constraints (list): A list (or nested list) where the first element is a list of constraints,
            provided in groups of 6 elements: [light_type, pos_x, pos_y, pos_z, min_intensity, max_intensity].

    Returns:
        list: A list of dictionaries, each containing the light's type, energy, location, and rotation.
    """
    # Create a World if one doesn't exist
    if bpy.context.scene.world is None:
        bpy.context.scene.world = bpy.data.worlds.new("World")
    bpy.context.scene.world.use_nodes = True
    bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[1].default_value = np.random.uniform(0.0, 1.0)

    # Process each group of 6 elements in lightning_position_constraints
    lightning_position_constraints_tuples = []
    for i in range(0, len(lightning_position_constraints[0]), 6):
        light_type = lightning_position_constraints[0][i]        # Light type (string)
        pos_x = float(lightning_position_constraints[0][i+1])      # X position
        pos_y = float(lightning_position_constraints[0][i+2])      # Y position
        pos_z = float(lightning_position_constraints[0][i+3])      # Z position
        min_intensity = int(lightning_position_constraints[0][i+4])# Minimum intensity
        max_intensity = int(lightning_position_constraints[0][i+5])# Maximum intensity

        lightning_position_constraints_tuples.append((light_type, pos_x, pos_y, pos_z, min_intensity, max_intensity))

    lights_info = []

    # Create a light for each constraint tuple
    for _, (light_type, light_x, light_y, light_z, min_intensity, max_intensity) in enumerate(lightning_position_constraints_tuples):
        bpy.ops.object.light_add(type=light_type, location=(light_x, light_y, light_z))
        light = bpy.context.object
        light.data.energy = float(random.randint(min_intensity, max_intensity))

        # Set light rotation angles (currently set to 0 for all axes)
        random_angle_x, random_angle_y, random_angle_z = 0, 0, 0

        light.rotation_euler = Euler(
            (radians(random_angle_x), radians(random_angle_y), radians(random_angle_z)), 'XYZ'
        )

        # Store light information in a dictionary
        lights_info.append({
            'type': light_type,
            'energy': f"{light.data.energy:.2f}",
            'location': f"({light_x:.2f}, {light_y:.2f}, {light_z:.2f})",
            'rotation': f"({radians(random_angle_x):.2f}, {radians(random_angle_y):.2f}, {radians(random_angle_z):.2f})"
        })

    return lights_info
