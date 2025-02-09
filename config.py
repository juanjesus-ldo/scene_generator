import yaml
import argparse

def load_config(yaml_path):
    """
    Load a YAML configuration file.

    Args:
        yaml_path (str): Path to the YAML file.

    Returns:
        dict: Parsed configuration.
    """
    with open(yaml_path, 'r') as file:
        return yaml.safe_load(file)

def validate_exclusive_args(config, arg1, arg2):
    """
    Ensure that exactly one of the two parameters is present in the configuration.

    Args:
        config (dict): The configuration dictionary.
        arg1 (str): The first parameter.
        arg2 (str): The second parameter.

    Raises:
        ValueError: If both or neither of the parameters are specified.
    """
    if (arg1 in config) + (arg2 in config) == 2:
        raise ValueError(f"Either '{arg1}' or '{arg2}' must be specified in the YAML file, but not both.")
    elif (arg1 in config) + (arg2 in config) == 0:
        raise ValueError(f"Either '{arg1}' or '{arg2}' must be specified in the YAML file.")

def validate_min_instances(min_instances, num_instances):
    """
    Validate that the minimum instances accepted is a positive integer and less than the total number of instances.

    Args:
        min_instances (int): Minimum accepted instances.
        num_instances (int): Total number of instances.

    Raises:
        argparse.ArgumentTypeError: If validation fails.
    """
    if min_instances < 1:
        raise argparse.ArgumentTypeError("Minimum instances accepted must be a positive integer.")
    if min_instances >= num_instances:
        raise argparse.ArgumentTypeError("Minimum instances accepted must be less than the total number of instances.")

def validate_camera_positions(values):
    """
    Validate camera positions by flattening the list (if nested), converting to floats,
    and ensuring the total number of values is a multiple of 3.

    Args:
        values (list): List (or nested list) of camera position values.

    Returns:
        list: Flattened list of camera position floats.

    Raises:
        argparse.ArgumentTypeError: If the number of values is not a multiple of 3.
    """
    # Flatten the list if it is a list of lists
    flat_values = [item for sublist in values for item in sublist] if any(isinstance(i, list) for i in values) else values

    float_values = [float(val) for val in flat_values]
    if len(float_values) % 3 != 0:
        raise argparse.ArgumentTypeError(f"Camera positions must be provided in multiples of 3, but {len(float_values)} values were provided.")
    return float_values

def validate_rotation_constraints(camera_positions, camera_rotation_constraints):
    """
    Validate that the number of rotation constraint tuples (each with 6 values)
    matches the number of camera position triplets.

    Args:
        camera_positions (list): Flattened list of camera positions.
        camera_rotation_constraints (list): List (or nested list) of rotation constraints.

    Raises:
        argparse.ArgumentTypeError: If the constraints are not provided in correct multiples.
    """
    flat_rotation_constraints = [item for sublist in camera_rotation_constraints for item in sublist] if any(isinstance(i, list) for i in camera_rotation_constraints) else camera_rotation_constraints

    num_camera_positions = len(camera_positions) / 3  # Number of camera triplets
    num_rotation_constraints = len(flat_rotation_constraints) / 6  # Number of 6-value tuples

    if (len(flat_rotation_constraints) % 6 == 0) and (num_rotation_constraints != num_camera_positions):
        raise argparse.ArgumentTypeError(
            f"The number of rotation constraints tuples ({int(num_rotation_constraints)}) does not match the number of camera positions tuples ({int(num_camera_positions)})."
        )
    elif len(flat_rotation_constraints) % 6 != 0:
        raise argparse.ArgumentTypeError(
            f"Rotation constraints must be provided in multiples of 6, but {len(flat_rotation_constraints)} values were provided."
        )

def validate_focal_length_constraints(camera_positions, focal_length_constraints):
    """
    Validate that the number of focal length constraint pairs (each with 2 values)
    matches the number of camera position triplets.

    Args:
        camera_positions (list): Flattened list of camera positions.
        focal_length_constraints (list): List (or nested list) of focal length constraints.

    Raises:
        argparse.ArgumentTypeError: If the constraints are not provided in correct multiples.
    """
    flat_focal_length_constraints = [item for sublist in focal_length_constraints for item in sublist] if any(isinstance(i, list) for i in focal_length_constraints) else focal_length_constraints

    num_camera_positions = len(camera_positions) / 3  # Number of camera triplets
    num_focal_length_constraints = len(flat_focal_length_constraints) / 2  # Number of pairs

    if (len(flat_focal_length_constraints) % 2 == 0) and (num_focal_length_constraints != num_camera_positions):
        raise argparse.ArgumentTypeError(
            f"The number of focal length constraint pairs ({int(num_focal_length_constraints)}) does not match the number of camera positions tuples ({int(num_camera_positions)})."
        )
    elif len(flat_focal_length_constraints) % 2 != 0:
        raise argparse.ArgumentTypeError(
            f"Focal length constraints must be provided in multiples of 2, but {len(flat_focal_length_constraints)} values were provided."
        )

def validate_lightning_constraints(lightning_constraints):
    """
    Validate the lightning constraints ensuring that:
      - The total number of values is a multiple of 6.
      - The light type is valid.
      - Position values (x, y, z) are floats.
      - Intensity values are integers, and the minimum is not greater than the maximum.

    Args:
        lightning_constraints (list): List (or nested list) of lightning constraints.

    Raises:
        ValueError: If any constraint does not meet the required format.
    """
    flat_constraints = [item for sublist in lightning_constraints for item in sublist] if any(isinstance(i, list) for i in lightning_constraints) else lightning_constraints

    if len(flat_constraints) % 6 != 0:
        raise ValueError("The number of elements in lightning_constraints must be a multiple of 6.")

    for i in range(0, len(flat_constraints), 6):
        light_type = flat_constraints[i]
        x, y, z = flat_constraints[i+1:i+4]
        min_intensity, max_intensity = flat_constraints[i+4:i+6]

        if light_type not in ["SUN", "POINT", "AREA", "SPOT"]:
            raise ValueError(f"Invalid light type '{light_type}' at position {i}. Must be 'SUN', 'POINT', 'AREA', or 'SPOT'.")

        try:
            x, y, z = float(x), float(y), float(z)
        except ValueError:
            raise ValueError(f"Position values (x, y, z) at positions {i+1}, {i+2}, {i+3} must be floats.")

        try:
            min_intensity, max_intensity = int(min_intensity), int(max_intensity)
        except ValueError:
            raise ValueError(f"Intensity values at positions {i+4}, {i+5} must be integers.")

        if min_intensity > max_intensity:
            raise ValueError(f"Min intensity ({min_intensity}) cannot be greater than max intensity ({max_intensity}) at positions {i+4}, {i+5}.")
