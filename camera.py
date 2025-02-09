import bpy
import random
import mathutils
import math

from objects import get_random_object

def init_camera(camera_position, camera_rotation_constraints, focal_length_constraints, obj, location=None, height=None):
    """
    Initialize a camera with a random position, rotation, and focal length.

    Parameters:
        camera_position: List of [x, y, z] triplets for possible camera positions.
        camera_rotation_constraints: List of lists containing 6 floats per camera [min_rot_x, max_rot_x, min_rot_y, max_rot_y, min_rot_z, max_rot_z].
        focal_length_constraints: List of lists containing 2 floats per camera [min_focal_length, max_focal_length].
        obj: Unused parameter.
        location: Optional location for camera placement (used with .blend files).
        height: Unused parameter.

    Returns:
        A tuple containing:
          - The created camera object.
          - The random index used for selecting camera constraints.
          - A formatted string of the camera position.
          - A formatted string of the camera rotation (in radians).
          - A formatted string of the camera focal length.
    """
    # Assume camera_position is already a list of [x, y, z] triplets
    camera_position_triplets = camera_position

    # Select a random camera position
    random_index = random.randint(0, len(camera_position_triplets) - 1)
    random_camera_position_triplet = camera_position_triplets[random_index]

    # Add the camera at the desired location
    if location is None:
        bpy.ops.object.camera_add(location=(random_camera_position_triplet[0],
                                              random_camera_position_triplet[1],
                                              random_camera_position_triplet[2]))
    else:
        bpy.ops.object.camera_add(location=location)

    camera = bpy.context.object

    # Convert camera rotation constraints into a list of tuples (each with 6 float values)
    camera_rotation_constraints_floats = [float(value) for value in sum(camera_rotation_constraints, [])]
    camera_rotation_constraints_tuples = [camera_rotation_constraints_floats[i:i + 6]
                                          for i in range(0, len(camera_rotation_constraints_floats), 6)]

    # Apply random rotations based on constraints corresponding to the selected index
    min_rot_x, max_rot_x, min_rot_y, max_rot_y, min_rot_z, max_rot_z = camera_rotation_constraints_tuples[random_index]
    camera.rotation_euler.x = math.radians(random.uniform(min_rot_x, max_rot_x))
    camera.rotation_euler.y = math.radians(random.uniform(min_rot_y, max_rot_y))
    camera.rotation_euler.z = math.radians(random.uniform(min_rot_z, max_rot_z))

    # Set the camera as the active scene camera
    bpy.context.scene.camera = camera

    # Convert focal length constraints into a list of pairs
    focal_length_constraints_floats = [float(value) for value in sum(focal_length_constraints, [])]
    focal_length_constraints_pairs = [focal_length_constraints_floats[i:i + 2]
                                      for i in range(0, len(focal_length_constraints_floats), 2)]

    # Set the camera's focal length within the specified range
    bpy.context.object.data.lens = random.uniform(focal_length_constraints_pairs[random_index][0],
                                                    focal_length_constraints_pairs[random_index][1])

    # Update the view layer to apply changes
    bpy.context.view_layer.update()

    return (
        camera,
        random_index,
        f"{random_camera_position_triplet[0]:.2f}, {random_camera_position_triplet[1]:.2f}, {random_camera_position_triplet[2]:.2f}",
        f"{camera.rotation_euler.x:.2f}, {camera.rotation_euler.y:.2f}, {camera.rotation_euler.z:.2f}",
        f"{bpy.context.object.data.lens:.2f}"
    )

def get_center(collection):
    """
    Calculate the centroid of all objects in the given collection.

    Parameters:
        collection: A Blender collection of objects.

    Returns:
        A mathutils.Vector representing the centroid of the collection.
    """
    total_location = mathutils.Vector((0, 0, 0))
    total_count = 0
    for obj in collection.objects:
        total_location += obj.location
        total_count += 1
    center = total_location / total_count
    return center

def get_closest_object_to_center(collection, center):
    """
    Find the object in the collection that is closest to the specified center.

    Parameters:
        collection: A Blender collection of objects.
        center: A mathutils.Vector representing the center point.

    Returns:
        The location of the object (as a mathutils.Vector) closest to the center.
    """
    closest_distance = float('inf')
    closest_object = None
    for obj in collection.objects:
        distance = (obj.location - center).length
        if distance < closest_distance:
            closest_distance = distance
            closest_object = obj
    return closest_object.location

def get_max_height(collection):
    """
    Calculate the maximum Z-coordinate (height) among the bounding boxes of objects in the collection.

    Parameters:
        collection: A Blender collection of objects.

    Returns:
        The maximum height value found.
    """
    max_height = -float('inf')
    for obj in collection.objects:
        bbox = obj.bound_box
        max_z = max(vertex[2] for vertex in bbox)
        max_height = max(max_height, max_z)
    return max_height

def get_collection_midpoint(collection):
    """
    Calculate the midpoint height among objects in the collection that share the maximum height.

    Parameters:
        collection: A Blender collection of objects.

    Returns:
        The average maximum height (midpoint) or None if the collection is empty or invalid.
    """
    max_height = get_max_height(collection)
    if max_height == -float('inf'):
        return None

    total_height = 0
    count = 0
    for obj in collection.objects:
        bbox = obj.bound_box
        max_z = max(vertex[2] for vertex in bbox)
        if max_z == max_height:
            total_height += max_z
            count += 1
    if count == 0:
        return None
    midpoint = total_height / count
    return midpoint

def adjust_camera(cubes):
    """
    Adjust the active camera so that it points toward a random object from the given list.

    Parameters:
        cubes: A list (or collection) of objects to select from.

    Effects:
        Modifies the camera's z-location and rotation to track the selected object.
    """
    camera = bpy.context.scene.camera
    look_at_obj = get_random_object(cubes)
    camera.location.z = random.uniform(0.3, camera.location.z)
    direction = look_at_obj.location - camera.location
    camera.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
