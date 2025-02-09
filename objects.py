import os
import random
import math
import bpy
import bmesh
import bpy_extras
import mathutils
import numpy as np

from mathutils.bvhtree import BVHTree
from utils import stdout_redirected, stderr_redirected, occlusion_test, intersection_check, intersection_objects

def random_rects(list_tam, list_height, maxtries, ply_obj, len_assets):
    """
    Generate random rectangles (with corresponding cube objects) within the bounding box of a given object.
    Ensures that generated rectangles do not overlap.

    Args:
        list_tam (list): List of tuples (width, depth) for each rectangle.
        list_height (list): List of heights for the rectangles.
        maxtries (int): Maximum number of attempts to generate a valid rectangle.
        ply_obj (Object): The object whose bounding box is used as a reference.
        len_assets (int): Number of asset objects (affects generation parameters).

    Returns:
        list: A list of rectangle dictionaries with updated positions.
    """
    class Rectangle:
        def __init__(self, x, y, z, w, h, angle_x=0, angle_y=0, angle_z=0):
            self.x = x
            self.y = y
            self.z = z
            self.width = w
            self.height = h
            self.angle_x = angle_x
            self.angle_y = angle_y
            self.angle_z = angle_z
            self.pts = None
            self.hpts = None
            self.lines = None

        def update_points(self):
            # Define a unit square centered at the origin for the base
            square = np.array([[-1, -1], [-1, 1], [1, 1], [1, -1], [-1, -1]])
            aux = square * (self.width / 2, self.height / 2)

            # Create rotation matrices for each axis
            rot_x = mathutils.Matrix.Rotation(math.radians(self.angle_x), 4, 'X')
            rot_y = mathutils.Matrix.Rotation(math.radians(self.angle_y), 4, 'Y')
            rot_z = mathutils.Matrix.Rotation(math.radians(self.angle_z), 4, 'Z')
            rot_final = rot_z @ rot_y @ rot_x

            # Apply rotation to base points and calculate points in 3D space
            aux_vectors = [mathutils.Vector((point[0], point[1], 0)) for point in aux]
            rotated_vectors = [rot_final @ vec for vec in aux_vectors]
            self.pts = np.array([[vec.x, vec.y, self.z] for vec in rotated_vectors]) + np.array([self.x, self.y, self.z])

    def intersection_check_objects(ok_objs_list, current_obj):
        """
        Check whether current_obj intersects with any object in ok_objs_list using BVH trees.

        Args:
            ok_objs_list (list): List of already placed objects.
            current_obj (Object): The object to check for intersections.

        Returns:
            bool: True if an intersection is found; False otherwise.
        """
        bm2 = bmesh.new()
        bm2.from_mesh(current_obj.data)
        bm2.transform(current_obj.matrix_world)
        obj_next_BVHtree = BVHTree.FromBMesh(bm2)

        for obj_now in ok_objs_list:
            bm1 = bmesh.new()
            bm1.from_mesh(obj_now.data)
            bm1.transform(obj_now.matrix_world)
            obj_now_BVHtree = BVHTree.FromBMesh(bm1)
            inter = obj_now_BVHtree.overlap(obj_next_BVHtree)
            if inter:
                return True
        return False

    def create_object(rect, index, h):
        """
        Create a cube object based on rectangle parameters, adjust its scale, rotation, and position.
        Uses ray casting to determine the proper ground level.

        Args:
            rect (dict): Dictionary with rectangle parameters.
            index (int): Index of the rectangle.
            h (float): Height dimension for the object.

        Returns:
            tuple: (cube object, updated rect dictionary) if successful; None otherwise.
        """
        x, y, z, w, depth = rect["x"], rect["y"], rect["z"], rect["width"], rect["depth"]
        angle_x, angle_y, angle_z = rect["angle_x"], rect["angle_y"], rect["angle_z"]

        bpy.ops.mesh.primitive_cube_add(size=1, location=(x, y, z))
        cube = bpy.context.object
        cube.scale = (depth, w, h)

        cube.select_set(True)
        bpy.context.view_layer.objects.active = cube
        cube.rotation_euler = (angle_x, angle_y, angle_z)

        def ray_cast_to_ground(obj, start_locations, ray_direction=(0, 0, -1)):
            """
            Cast rays downward from multiple start locations to determine the ground level.

            Args:
                obj (Object): The object from which to cast rays.
                start_locations (list): List of starting locations (vectors).
                ray_direction (tuple): Direction vector for the ray.

            Returns:
                float: The minimum z-value among hit locations, or None if no hit is found.
            """
            scene = bpy.context.scene
            depsgraph = bpy.context.evaluated_depsgraph_get()
            ray_direction = mathutils.Vector(ray_direction)

            # Hide the object to avoid interference with ray casting
            obj.hide_viewport = True
            z_values = []
            for location in start_locations:
                result, hit_location, _, _, _, _ = scene.ray_cast(depsgraph, location, ray_direction)
                if result:
                    z_values.append(hit_location.z)
            obj.hide_viewport = False
            return min(z_values) if z_values else None

        # Adjust height based on object type (boxes vs. asset objects)
        if index <= len(list_tam) - len_assets:  # For boxes
            if angle_y in [math.pi / 2, 3 * math.pi / 2]:
                h = (depth / 2) + 0.015
            elif (angle_x in [0.0, math.pi]) and (angle_y in [0.0, math.pi]):
                h = (h / 2) + 0.015
            elif (angle_x in [math.pi / 2, 3 * math.pi / 2]) and (angle_y in [0.0, math.pi]):
                h = (w / 2) + 0.015
        else:  # For asset objects
            if angle_y in [math.pi / 2, 3 * math.pi / 2]:
                h = (w / 2) + 0.01
            elif (angle_x in [math.pi / 2]) and (angle_y in [0.0]):
                h = (depth / 2) + 0.01

        vertices = [cube.matrix_world @ v.co for v in cube.data.vertices]
        start_locations = vertices + [cube.location]
        new_z = ray_cast_to_ground(cube, start_locations)

        if new_z is None:
            bpy.data.objects.remove(cube, do_unlink=True)
            return None
        else:
            bpy.ops.transform.translate(value=(x, y, new_z + h))
            rect["z"] = new_z + h

        return cube, rect

    def generate_rectangle(width, depth, index, maxtries=100):
        """
        Generate a rectangle within the bounding box of ply_obj that is visible from the camera.

        Args:
            width (float): Width of the rectangle.
            depth (float): Depth of the rectangle.
            index (int): Index of the rectangle.
            maxtries (int): Maximum number of attempts to generate a valid rectangle.

        Returns:
            dict: A dictionary with rectangle parameters if successful; None otherwise.
        """
        scene = bpy.context.scene
        cam = scene.camera

        # Get bounding box of the ply_obj
        bbox = [ply_obj.matrix_world @ mathutils.Vector(corner) for corner in ply_obj.bound_box]
        min_x = min(v.x for v in bbox)
        max_x = max(v.x for v in bbox)
        min_y = min(v.y for v in bbox)
        max_y = max(v.y for v in bbox)
        min_z = min(v.z for v in bbox)

        for _ in range(maxtries):
            # Generate a random point within the object's bounding box on the ground plane (z = min_z)
            rand_x = random.uniform(min_x, max_x)
            rand_y = random.uniform(min_y, max_y)
            rect_position = mathutils.Vector((rand_x, rand_y, min_z))

            if index <= len(list_tam) - len_assets:
                angle_x = math.radians(random.choice([0, 90, 180, 270]))
                angle_y = math.radians(random.choice([0, 90, 180, 270]))
                angle_z = math.radians(random.uniform(0, 360))
            else:
                angle_x = math.radians(90)
                angle_y = math.radians(random.choice([0, 90, 270]))
                angle_z = math.radians(random.uniform(0, 360))

            rotation_matrix = mathutils.Euler((angle_x, angle_y, angle_z)).to_matrix().to_4x4()
            rotated_position = rotation_matrix @ mathutils.Vector((rect_position.x, rect_position.y, rect_position.z))
            rect_position.x, rect_position.y, rect_position.z = rotated_position.x, rotated_position.y, rotated_position.z

            co_in_camera = bpy_extras.object_utils.world_to_camera_view(scene, cam, rect_position)
            if 0 <= co_in_camera.x <= 1 and 0 <= co_in_camera.y <= 1 and co_in_camera.z >= 0:
                return {
                    'x': rect_position.x,
                    'y': rect_position.y,
                    'z': rect_position.z,
                    'width': width,
                    'depth': depth,
                    'angle_x': angle_x,
                    'angle_y': angle_y,
                    'angle_z': angle_z
                }
        return None

    rects = []
    index = 0
    ok_objs_list = []
    for w_z, h in zip(list_tam, list_height):
        w, z = w_z
        index += 1
        for _ in range(maxtries):
            new_rect = generate_rectangle(w, z, index, maxtries)
            if new_rect is not None:
                try:
                    result = create_object(new_rect, index, h)
                    if result is None:
                        continue
                    obj, new_rect = result
                except Exception:
                    continue
                if not rects:
                    rects.append(new_rect)
                    ok_objs_list.append(obj)
                    break
                else:
                    if not intersection_check_objects(ok_objs_list, obj):
                        rects.append(new_rect)
                        ok_objs_list.append(obj)
                        break
                    else:
                        bpy.data.objects.remove(bpy.data.objects[obj.name], do_unlink=True)
    for obj in ok_objs_list:
        bpy.data.objects.remove(bpy.data.objects[obj.name], do_unlink=True)
    del ok_objs_list

    return rects


def create_prism(tb, rect, h, scene_obj, config_yaml):
    """
    Create a textured prism (cube) based on the provided rectangle parameters.
    Applies texture mapping to each face of the cube.

    Args:
        tb (str): Box type identifier.
        rect (dict): Dictionary with rectangle parameters (position, dimensions, rotation).
        h (float): Height dimension for the prism.
        scene_obj: Scene object (unused in this function).
        config_yaml (dict): Configuration dictionary containing asset paths.

    Returns:
        Object: The created cube object with applied textures.
    """
    x, y, z = rect["x"], rect["y"], rect["z"]
    w, depth = rect["width"], rect["depth"]
    angle_x, angle_y, angle_z = rect["angle_x"], rect["angle_y"], rect["angle_z"]

    bpy.ops.mesh.primitive_cube_add(size=1, location=(x, y, z))
    cube = bpy.context.object
    cube["BoxType"] = tb
    cube.scale = (depth, w, h)
    cube.select_set(True)
    bpy.context.view_layer.objects.active = cube
    cube.rotation_euler = (angle_x, angle_y, angle_z)

    face_images = [
        "Frontal_" + tb + ".jpeg",
        "Lateral_" + tb + ".jpeg",
        "Frontal_Rear_" + tb + ".jpeg",
        "Lateral_Rear_" + tb + ".jpeg",
        "Top_Rear_" + tb + ".jpeg",
        "Top_" + tb + ".jpeg"
    ]

    # Apply texture mapping for each prism face
    for i, texture in enumerate(face_images):
        img = None
        texture_path = os.path.join(config_yaml["dictionary_cubes"], tb, texture)
        if os.path.isfile(texture_path):
            img = bpy.data.images.load(texture_path)
        if img is None:
            print(f"No valid texture found in {texture_path}. Exiting...")
            sys.exit(0)

        # Create material and set up its node tree
        mat = bpy.data.materials.new(name="Material")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        # Remove all existing nodes
        for node in list(nodes):
            nodes.remove(node)
        # Create the texture image node, Principled BSDF node, and Material Output node
        texture_node = nodes.new("ShaderNodeTexImage")
        texture_node.image = img
        bsdf_node = nodes.new("ShaderNodeBsdfPrincipled")
        output_node = nodes.new("ShaderNodeOutputMaterial")
        # Link the texture node's Color output to the Base Color input of the Principled BSDF node
        links.new(bsdf_node.inputs['Base Color'], texture_node.outputs['Color'])
        # Link the BSDF node's BSDF output to the Surface input of the Material Output node
        links.new(output_node.inputs['Surface'], bsdf_node.outputs['BSDF'])
        # Append the new material to the cube's materials
        cube.data.materials.append(mat)
        cube.data.polygons[i].material_index = len(cube.data.materials) - 1

    # Set UV projection on each face
    cube.select_set(True)
    bpy.context.view_layer.objects.active = cube
    bpy.ops.object.mode_set(mode='EDIT')
    for face in cube.data.polygons:
        face.select = True
        bpy.ops.uv.cube_project(cube_size=h, correct_aspect=False, clip_to_bounds=False, scale_to_bounds=True)
        face.select = False
    bpy.ops.object.mode_set(mode='OBJECT')

    # Adjust UV coordinates for lateral faces
    for i, face in enumerate(cube.data.polygons):
        if i in [0, 1, 2, 3]:
            for loop_index in face.loop_indices:
                uv_coords = cube.data.uv_layers.active.data[loop_index].uv
                uv_coords[0] = 1.0 - uv_coords[0]

    return cube


def create_scene_objects(rnd_prisms, assets_files, scene_obj, config_yaml):
    """
    Create objects for the scene either by generating prisms or by importing asset objects.
    Guarantees that every created object has the "BoxType" property.

    Args:
        rnd_prisms (list): List of tuples (box type, rectangle, height).
        assets_files (list): List of asset file identifiers.
        scene_obj: Scene object (unused in this function).
        config_yaml (dict): Configuration dictionary with asset paths.

    Returns:
        list: List of created objects.
    """
    cubes = []
    for boxtype, rect, h in rnd_prisms:
        if boxtype not in assets_files:
            cube = create_prism(boxtype, rect, h, scene_obj, config_yaml)
            if cube is not None:
                if "BoxType" not in cube:
                    cube["BoxType"] = boxtype
                cubes.append(cube)
        else:
            filepath = os.path.join(config_yaml["assets"], boxtype + ".obj")
            with stdout_redirected(), stderr_redirected():
                bpy.ops.wm.obj_import(filepath=filepath)
            asset_obj = bpy.context.selected_objects[0]
            asset_obj["BoxType"] = boxtype
            x, y, z = rect["x"], rect["y"], rect["z"]
            angle_x, angle_y, angle_z = rect["angle_x"], rect["angle_y"], rect["angle_z"]
            asset_obj.location = (x, y, z)
            asset_obj.select_set(True)
            bpy.context.view_layer.objects.active = asset_obj
            asset_obj.rotation_euler = (angle_x, angle_y, angle_z)
            cubes.append(asset_obj)
    return cubes


def remove_invisible_objects(cubes):
    """
    Remove objects from the scene that are not visible from the camera.

    Returns:
        list: List of objects that remain visible.
    """
    res_ratio = 0.5
    res_x = int(bpy.context.scene.render.resolution_x * res_ratio)
    res_y = int(bpy.context.scene.render.resolution_y * res_ratio)
    visible_objs = occlusion_test(
        bpy.context.scene,
        bpy.context.evaluated_depsgraph_get(),
        bpy.context.scene.objects['Camera'],
        res_x, res_y
    )
    invisible_names = {o.name for o in bpy.context.scene.objects if o.type == 'MESH' and o.name not in visible_objs}
    visible_cubes = []
    for cube in cubes:
        if cube.name in invisible_names:
            bpy.data.objects.remove(bpy.data.objects[cube.name], do_unlink=True)
        else:
            visible_cubes.append(cube)
    return visible_cubes


def remove_intersections(cubes, obj_ply_BVHtree):
    """
    Remove objects that intersect with the main scene (using a BVH tree).

    Args:
        cubes (list): List of objects.
        obj_ply_BVHtree: BVH tree of the main ply object.

    Returns:
        list: Filtered list of objects.
    """
    cubes = intersection_check(cubes, obj_ply_BVHtree)
    return cubes


def remove_object_intersections(cubes):
    """
    Remove objects that intersect with each other.

    Args:
        cubes (list): List of objects.

    Returns:
        list: List of objects without mutual intersections.
    """
    valid_cubes = []
    for cube in cubes:
        if intersection_objects(valid_cubes, cube):
            bpy.data.objects.remove(bpy.data.objects[cube.name], do_unlink=True)
        else:
            valid_cubes.append(cube)
    return valid_cubes


def get_random_object(instances_ok):
    """
    Select a random object of type MESH from the provided list.

    Args:
        instances_ok (list): List of mesh objects.

    Returns:
        Object: A randomly selected object.
    """
    return random.choice(instances_ok)
