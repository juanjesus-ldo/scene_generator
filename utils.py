import os
import sys
import bpy
import random
import numpy as np
import shutil
import cv2
import time
import bmesh
import OpenEXR
import Imath
import bpy_extras
import csv

from contextlib import contextmanager
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from tabulate import tabulate

@contextmanager
def stdout_redirected(to=os.devnull):
    """
    Context manager to redirect stdout to the specified file or device.

    Example:
        with stdout_redirected(to="output.txt"):
            print("This goes to output.txt")
    """
    fd = sys.stdout.fileno()

    def _redirect_stdout(to):
        sys.stdout.close()  # Implicitly flush
        os.dup2(to.fileno(), fd)
        sys.stdout = os.fdopen(fd, 'w')

    with os.fdopen(os.dup(fd), 'w') as old_stdout:
        with open(to, 'w') as file:
            _redirect_stdout(to=file)
        try:
            yield
        finally:
            _redirect_stdout(to=old_stdout)

@contextmanager
def stderr_redirected(to=os.devnull):
    """
    Context manager to redirect stderr to the specified file or device.

    Example:
        with stderr_redirected(to="error_log.txt"):
            raise Exception("This error will be written to error_log.txt")
    """
    fd = sys.stderr.fileno()

    def _redirect_stderr(to):
        sys.stderr.close()
        os.dup2(to.fileno(), fd)
        sys.stderr = os.fdopen(fd, 'w')

    with os.fdopen(os.dup(fd), 'w') as old_stderr:
        with open(to, 'w') as file:
            _redirect_stderr(to=file)
        try:
            yield
        finally:
            _redirect_stderr(to=old_stderr)

def read_exr_single_channel(image_path, channel_='V'):
    """
    Read a single-channel EXR image and return it as a NumPy array.

    Args:
        image_path (str): Path to the EXR file.
        channel_ (str): Channel name to read (default 'V').

    Returns:
        ndarray: The image data as a float32 NumPy array.
    """
    exr_file = OpenEXR.InputFile(image_path)
    header = exr_file.header()
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1
    channel_data = exr_file.channel(channel_, Imath.PixelType(Imath.PixelType.FLOAT))
    img = np.frombuffer(channel_data, dtype=np.float32).reshape((height, width))
    return img

def save_exr_single_channel(image_path, img, channel_name='V'):
    """
    Save a single-channel image (NumPy array) as an EXR file.

    Args:
        image_path (str): Output EXR file path.
        img (ndarray): Image data as a NumPy array.
        channel_name (str): Name of the channel.
    """
    header = OpenEXR.Header(img.shape[1], img.shape[0])
    header['channels'] = {channel_name: Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))}
    exr_file = OpenEXR.OutputFile(image_path, header)
    exr_file.writePixels({channel_name: img.tobytes()})
    exr_file.close()

def point_within_image(x, y, img_width, img_height):
    """Check if a point (x, y) is within the image bounds."""
    return 0 <= x < img_width and 0 <= y < img_height

def any_point_within_image(points, img_width, img_height):
    """Check if any point in a list is within the image bounds."""
    return any(point_within_image(x, y, img_width, img_height) for x, y in points)

def world_to_camera_view_own(scene, obj, coord, camera, matrix_world):
    """
    Project a 3D coordinate into screen space (pixel coordinates).

    Args:
        scene: The current scene.
        obj: The object (unused in projection but may be useful for context).
        coord (Vector): 3D coordinate.
        camera: The camera object.
        matrix_world: World transformation matrix of the object.

    Returns:
        tuple: (screen_x, screen_y)
    """
    co_local = matrix_world @ coord  # Object local to world
    co_ndc = bpy_extras.object_utils.world_to_camera_view(scene, camera, co_local)
    render_scale = scene.render.resolution_percentage / 100
    render_size = (scene.render.resolution_x * render_scale,
                   scene.render.resolution_y * render_scale)
    screen_x = co_ndc.x * render_size[0]
    screen_y = (1.0 - co_ndc.y) * render_size[1]
    return screen_x, screen_y

def is_bbox_behind_ply(scene, camera, obj, ply_obj, depsgraph):
    """
    Check if all corners of an object's bounding box are behind the ply_obj (using ray casting).

    Args:
        scene: The current scene.
        camera: The camera object.
        obj: The object to test.
        ply_obj: The reference object.
        depsgraph: The dependency graph.

    Returns:
        bool: True if all corners are behind ply_obj, False otherwise.
    """
    matrix_world = obj.matrix_world
    bbox_corners = [matrix_world @ Vector(corner) for corner in obj.bound_box]
    for corner in bbox_corners:
        origin = camera.location
        direction = (corner - origin).normalized()
        hit, location, normal, face_index, hit_obj, matrix = scene.ray_cast(depsgraph, origin, direction)
        if hit and hit_obj == ply_obj:
            continue
        else:
            return False
    return True

def get_rotation_translation(obj):
    """
    Get the translation and rotation (as a 3x3 matrix) from an object's world matrix.

    Args:
        obj: The object.

    Returns:
        tuple: (translation (Vector), rotation (Matrix))
    """
    matrix_world = obj.matrix_world
    translation = matrix_world.to_translation()
    rotation = matrix_world.to_3x3()
    return translation, rotation

def add_object_to_image(image_id, obj_name, boxtype, cam_R_m2c, cam_t_m2c, focal_length, principal_point, points_pixels, data):
    """
    Add object information (including screen-space bounding box points) to a JSON structure.

    Args:
        image_id (str): Identifier for the image.
        obj_name (str): Name of the object.
        boxtype (str): Object type.
        cam_R_m2c: Rotation matrix (list of lists).
        cam_t_m2c: Translation vector (list).
        focal_length: Focal length.
        principal_point: Principal point.
        points_pixels: List of 2D points.
        data (dict): JSON structure to update.
    """
    image = data["images"].get(image_id)
    if image is None:
        image = {"intrinsic": [], "objects": []}
        data["images"][image_id] = image
        new_intrinsic_params = {
            "focal_length_mm": focal_length,
            "principal_point": principal_point,
        }
        image["intrinsic"].append(new_intrinsic_params)
    new_object = {
        "points_pixels": list(points_pixels),
        "cam_R_m2c": [r for row in cam_R_m2c for r in row],
        "cam_t_m2c": list(cam_t_m2c),
        "obj_id": obj_name,
        "obj_type": boxtype
    }
    image["objects"].append(new_object)

def intersection_check(objs_list, obj_ply_BVHtree):
    """
    Remove objects from a list that intersect with a given BVH tree.

    Args:
        objs_list (list): List of objects.
        obj_ply_BVHtree: BVH tree of the reference object.

    Returns:
        list: List of objects that do not intersect with the reference.
    """
    ok_objs_list = []
    for obj_now in objs_list:
        bm1 = bmesh.new()
        bm1.from_mesh(obj_now.data)
        bm1.transform(obj_now.matrix_world)
        obj_now_BVHtree = BVHTree.FromBMesh(bm1)
        inter = obj_now_BVHtree.overlap(obj_ply_BVHtree)
        if inter != []:
            bpy.data.objects.remove(bpy.data.objects[obj_now.name], do_unlink=True)
        else:
            ok_objs_list.append(obj_now)
    return ok_objs_list

def intersection_objects(ok_objs_list, current_obj):
    """
    Check if the current object intersects with any object in the provided list.

    Args:
        ok_objs_list (list): List of objects.
        current_obj: The object to test.

    Returns:
        bool: True if an intersection is found, False otherwise.
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
        if inter != []:
            return True
    return False

def generate_unique_color(used_colors):
    """
    Generate a unique random RGB color not in the used_colors set.

    Args:
        used_colors (set): Set of colors already used.

    Returns:
        tuple: A unique random color (R, G, B) with values between 0 and 1.
    """
    while True:
        color = (random.random(), random.random(), random.random())
        if color not in used_colors:
            used_colors.add(color)
            return color

def find_bounding_box(image, pixel_tuple):
    """
    Find the bounding box (min and max x, y coordinates) for pixels matching a given value.

    Args:
        image (ndarray): Image as a NumPy array.
        pixel_tuple (tuple): Pixel value to search for.

    Returns:
        tuple or None: (x_min, y_min, x_max, y_max) or None if no match.
    """
    matches = image == pixel_tuple[0]  # For grayscale images
    indices = np.argwhere(matches)
    if len(indices) == 0:
        return None
    x_min = np.min(indices[:, 1])
    y_min = np.min(indices[:, 0])
    x_max = np.max(indices[:, 1])
    y_max = np.max(indices[:, 0])
    return x_min, y_min, x_max, y_max

def occlusion_test(scene, depsgraph, camera, resolution_x, resolution_y):
    """
    Perform an occlusion test by casting rays through a grid over the camera view.

    Args:
        scene: The current scene.
        depsgraph: The dependency graph.
        camera: The camera object.
        resolution_x (int): Horizontal resolution for sampling.
        resolution_y (int): Vertical resolution for sampling.

    Returns:
        set: Set of names of objects that are hit by the rays.
    """
    top_right, _, bottom_left, top_left = camera.data.view_frame(scene=scene)
    camera_quaternion = camera.matrix_world.to_quaternion()
    camera_translation = camera.matrix_world.translation
    x_range = np.linspace(top_left[0], top_right[0], resolution_x)
    y_range = np.linspace(top_left[1], bottom_left[1], resolution_y)
    z_dir = top_left[2]
    hit_data = set()
    for x in x_range:
        for y in y_range:
            pixel_vector = Vector((x, y, z_dir))
            pixel_vector.rotate(camera_quaternion)
            pixel_vector.normalized()
            is_hit, _, _, _, hit_obj, _ = scene.ray_cast(depsgraph, camera_translation, pixel_vector)
            if is_hit:
                hit_data.add(hit_obj.name)
    return hit_data

def get_scene_bbox():
    """
    Compute the bounding box of the entire scene.

    Returns:
        tuple: ((min_x, min_y, min_z), (max_x, max_y, max_z))
    """
    scene = bpy.context.scene
    bbox_corners = [scene.objects[obj.name].matrix_world @ Vector(corner)
                    for obj in scene.objects for corner in obj.bound_box]
    min_x = min(bbox_corners, key=lambda v: v[0])[0]
    max_x = max(bbox_corners, key=lambda v: v[0])[0]
    min_y = min(bbox_corners, key=lambda v: v[1])[1]
    max_y = max(bbox_corners, key=lambda v: v[1])[1]
    min_z = min(bbox_corners, key=lambda v: v[2])[2]
    max_z = max(bbox_corners, key=lambda v: v[2])[2]
    return (min_x, min_y, min_z), (max_x, max_y, max_z)

def set_tex_cube_flat(cube, color):
    """
    Replace the cube's materials with a flat-color emission shader for segmentation.

    Args:
        cube: The cube object.
        color (tuple): RGB color as a tuple; an alpha value is added.
    """
    cube_material_flat = bpy.data.materials.new(name="Cube_Material_Flat")
    cube_material_flat.use_nodes = True
    nodes = cube_material_flat.node_tree.nodes
    nodes.clear()
    emission = nodes.new(type='ShaderNodeEmission')
    # Append alpha value (1) to the color tuple
    emission.inputs[0].default_value = color + (1,)
    output = nodes.new(type='ShaderNodeOutputMaterial')
    cube_material_flat.node_tree.links.new(emission.outputs[0], output.inputs[0])
    cube.data.materials[0] = cube_material_flat
    for i in range(6):
        cube.data.polygons[i].material_index = 0

def clean_objects_from_scene(objects):
    """
    Remove the specified objects from the scene.

    Args:
        objects (list): List of objects to remove.
    """
    for obj in objects:
        bpy.data.objects.remove(obj, do_unlink=True)

def clean_blender():
    """
    Reinitialize Blender by loading the default home file.
    """
    bpy.ops.wm.read_homefile(use_empty=True)

def clean_blender_keep_ply(ply_object_name):
    """
    Clean the scene by deleting all objects except the specified PLY object.

    Args:
        ply_object_name (str): Name of the PLY object to keep.
    """
    ply_object = bpy.data.objects.get(ply_object_name)
    if ply_object is None:
        print(f"Object '{ply_object_name}' not found.")
        return
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.context.scene.objects:
        if obj.name != ply_object_name:
            obj.select_set(True)
    bpy.ops.object.delete()
    with stdout_redirected():
        bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)
    bpy.context.view_layer.objects.active = ply_object
    ply_object.select_set(True)

def save_ply_object_data(obj_name):
    """
    Save data of a PLY object.

    Args:
        obj_name (str): Name of the object.

    Returns:
        dict or None: Dictionary with object data or None if not found.
    """
    obj = bpy.data.objects.get(obj_name)
    if obj is None:
        print(f"Object '{obj_name}' not found.")
        return None
    obj_data = {
        'name': obj.name,
        'location': obj.location.copy(),
        'rotation_euler': obj.rotation_euler.copy(),
        'scale': obj.scale.copy(),
        'mesh': obj.data.copy()
    }
    return obj_data

def restore_ply_object(obj_data):
    """
    Restore a PLY object from saved data.

    Args:
        obj_data (dict): Dictionary containing object data.
    """
    if obj_data is None:
        print("No data to restore the object.")
        return
    mesh = obj_data['mesh']
    new_obj = bpy.data.objects.new(obj_data['name'], mesh)
    new_obj.location = obj_data['location']
    new_obj.rotation_euler = obj_data['rotation_euler']
    new_obj.scale = obj_data['scale']
    bpy.context.collection.objects.link(new_obj)

def export_ply_object(obj_name, export_path):
    """
    Export a PLY object to a file.

    Args:
        obj_name (str): Name of the object.
        export_path (str): File path for export.
    """
    bpy.ops.object.select_all(action='DESELECT')
    obj = bpy.data.objects.get(obj_name)
    if obj:
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.export_mesh.ply(filepath=export_path)
    else:
        print(f"Object '{obj_name}' not found.")

def import_ply_object(import_path):
    """
    Import a PLY object from a file.

    Args:
        import_path (str): File path to the PLY file.
    """
    bpy.ops.import_mesh.ply(filepath=import_path)

def find_key(dictionary, a, b, c):
    """
    Find the key in the dictionary whose value is closest (by sum of absolute values) to the sum of a, b, c.

    Args:
        dictionary (dict): Dictionary with numeric values.
        a, b, c (numbers): Values to compare.

    Returns:
        key: The key with the minimal difference.
    """
    min_difference = float('inf')
    min_key = None
    for key, value in dictionary.items():
        sum_absolute_values = sum(abs(value[i]) for i in range(len(value)))
        sum_abc = abs(a) + abs(b) + abs(c)
        difference = abs(sum_absolute_values - sum_abc)
        if difference < min_difference:
            min_difference = difference
            min_key = key
    return min_key

def get_obj_dimensions(filepath):
    """
    Import an .obj file and return its dimensions.

    Args:
        filepath (str): Path to the .obj file.

    Returns:
        Vector: Dimensions of the imported object.
    """
    with stdout_redirected(), stderr_redirected():
        bpy.ops.wm.obj_import(filepath=filepath)
    imported_object = bpy.context.selected_objects[0]
    dimensions = imported_object.dimensions
    bpy.ops.object.select_all(action='DESELECT')
    bpy.data.objects.remove(imported_object, do_unlink=True)
    return dimensions

def set_render_dimensions(width, height):
    """Set the render resolution of the scene."""
    bpy.context.scene.render.resolution_x = width
    bpy.context.scene.render.resolution_y = height

def initialize_stats(prism_types, assets_obj):
    """
    Initialize statistics dictionaries for each type.

    Args:
        prism_types (list): List of prism type names.
        assets_obj (list): List of asset object identifiers.

    Returns:
        tuple: Two dictionaries (accepted and rejected counts) keyed by type.
    """
    stats_aux = {}
    stats_rej_aux = {}
    for key in prism_types + assets_obj:
        key_name = key[:-4] if key.endswith('.obj') else key
        stats_aux[key_name] = 0
        stats_rej_aux[key_name] = 0
    return stats_aux, stats_rej_aux

def select_object_file(obj_files):
    """
    Select a random object file and determine its type (.obj, .blend, or .ply).

    Args:
        obj_files (list): List of file names.

    Returns:
        tuple: (selected file, obj_f flag, blend_f flag, ply_f flag)
    """
    obj_file = random.choice(obj_files)
    obj_f = obj_file.endswith(".obj")
    blend_f = obj_file.endswith(".blend")
    ply_f = obj_file.endswith(".ply")
    return obj_file, obj_f, blend_f, ply_f

def split_prism_types(rnd_prisms_types):
    """
    Split prism types into asset and non-asset lists based on file extension.

    Args:
        rnd_prisms_types (list): List of prism type identifiers.

    Returns:
        tuple: (assets_files, non_obj_files)
    """
    assets_files = [f[:-4] for f in rnd_prisms_types if f.endswith('.obj')]
    non_obj_files = [f for f in rnd_prisms_types if not f.endswith('.obj')]
    return assets_files, non_obj_files

def get_dimensions(files, dims_dict):
    """
    Get the base (width, depth) and height dimensions for a list of files using a dictionary.

    Args:
        files (list): List of file identifiers.
        dims_dict (dict): Dictionary mapping file identifiers to dimensions.

    Returns:
        tuple: (base_dims, height_dims)
    """
    base_dims = [(dims_dict[f][0], dims_dict[f][2]) for f in files]
    height_dims = [dims_dict[f][1] for f in files]
    return base_dims, height_dims

def compute_bounding_boxes(instances_fn, classes_fn, scene_shading, cubes, config_yaml,
                           boxes_types, data_json, img_id, stats_boxes_aux, stats_boxes_rej_aux):
    """
    Compute 2D bounding boxes from the instance mask and update the JSON structure.

    Args:
        instances_fn (str): Path to the instance mask EXR file.
        classes_fn (str): Path to the class mask EXR file.
        scene_shading (ndarray): Shading image as a NumPy array.
        cubes (list): List of scene objects.
        config_yaml (dict): Configuration dictionary.
        boxes_types (list): List of box type names.
        data_json (dict): JSON structure for storing object data.
        img_id (int): Image identifier.

    Returns:
        tuple: (accepted bounding boxes, rejected bounding boxes, rejected cubes, remaining cubes)
    """
    class_img = read_exr_single_channel(classes_fn)
    inst_img = read_exr_single_channel(instances_fn)
    height, width = inst_img.shape
    pixel_tuples = inst_img.reshape((height * width, 1))
    unique_pixels, counts = np.unique(pixel_tuples, axis=0, return_counts=True)
    instances_rgb_dict = dict(zip([tuple(pixel) for pixel in unique_pixels], counts))
    bbxs, bbxs_rej = [], []
    cubes_rejected, cubes_remaining = [], []

    for pixel_rgb in instances_rgb_dict:
        mask_img_inst = (inst_img == pixel_rgb[0]).astype('uint8')
        result_img = inst_img * mask_img_inst.astype(np.uint8)
        index_obj_inst = np.unique(result_img)[1:]
        if len(index_obj_inst) > 0 and index_obj_inst != 0:
            coords = np.argwhere(result_img == index_obj_inst)
            r, c = coords[0][0], coords[0][1]
            index_obj_class = class_img[r, c]
            box_type = boxes_types[int(index_obj_class) - 1]
        else:
            continue
        bbox = find_bounding_box(inst_img, pixel_rgb)
        if bbox is None:
            continue
        x_min, y_min, x_max, y_max = bbox
        img_key = f"{img_id:05}"
        if img_key in data_json["images"]:
            data_json["images"][img_key]["objects"] = [
                {**obj, "bbox": [int(x_min), int(y_min), int(x_max), int(y_max)]}
                if obj["obj_id"] == cubes[int(index_obj_inst[0]) - 1].name else obj
                for obj in data_json["images"][img_key]["objects"]
            ]
        if (x_max - x_min + 1) * (y_max - y_min + 1) == height * width:
            continue
        sub_img = np.array(result_img[y_min:y_max + 1, x_min:x_max + 1])
        sub_img = np.stack([sub_img] * 3, axis=-1)
        sub_img_flat = sub_img.reshape(-1, 3)
        try:
            terna_rgb = np.unique(sub_img_flat, axis=0)[1:][0]
        except:
            terna_rgb = np.unique(sub_img_flat, axis=0)[0]
        valor_pixel = np.array(terna_rgb).reshape(1, 1, 3)
        diff = np.abs(sub_img - valor_pixel)
        sum_diff = np.sum(diff, axis=2)
        umbral = 0.01
        pix_count_instance = np.sum(sum_diff < umbral)
        box_cvr = int(100 * pix_count_instance / ((x_max - x_min + 1) * (y_max - y_min + 1)))
        bbx = {"img_name": f"{img_id:05}.PNG",
               "cls_name": box_type,
               "cls_key": index_obj_class,
               "ins_key": index_obj_inst[0],
               "x_min": x_min,
               "y_min": y_min,
               "x_max": x_max,
               "y_max": y_max,
               "w": x_max - x_min + 1,
               "h": y_max - y_min + 1,
               "cvr": box_cvr}
        min_w = int(config_yaml["min_width_bbox"] * bpy.context.scene.render.resolution_x / 100)
        min_h = int(config_yaml["min_height_bbox"] * bpy.context.scene.render.resolution_y / 100)
        if (bbx["w"] > min_w) and (bbx["h"] > min_h) and (bbx["cvr"] > config_yaml["min_box_coverage"]):
            bbxs.append(bbx)
            stats_boxes_aux[bbx["cls_name"]] += 1
            cubes_remaining.append(cubes[int(index_obj_inst[0]) - 1])
        else:
            bbxs_rej.append(bbx)
            cubes_rejected.append(cubes[int(index_obj_inst[0]) - 1])
            stats_boxes_rej_aux[bbx["cls_name"]] += 1
    return bbxs, bbxs_rej, cubes_rejected, cubes_remaining

def render_scene(outfile, bbg):
    """
    Render the current scene to the specified output file.
    If 'bbg' is True, set the background color to black.

    Args:
        outfile (str): Output file path.
        bbg (bool): If True, set background to black.
    """
    if bbg:
        if bpy.context.scene.world is None:
            bpy.context.scene.world = bpy.data.worlds.new("World")
        bpy.context.scene.world.use_nodes = True
        bg = bpy.context.scene.world.node_tree.nodes['Background']
        bg.inputs[0].default_value = (0, 0, 0, 1)
    bpy.context.scene.render.filepath = outfile
    with stdout_redirected():
        bpy.ops.render.render(write_still=True)

def render_depth_map(cur_base_path, img_id):
    """
    Render and process the depth map, then save it as an EXR file.

    Args:
        cur_base_path (str): Base directory for outputs.
        img_id (int): Image ID.

    Returns:
        str: Path to the saved depth map.
    """
    bpy.context.scene.view_layers["ViewLayer"].use_pass_z = True
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    for n in tree.nodes:
        tree.nodes.remove(n)
    rl = tree.nodes.new('CompositorNodeRLayers')
    fileOutput = tree.nodes.new(type="CompositorNodeOutputFile")
    fileOutput.base_path = "/tmp/depth"
    fileOutput.format.file_format = 'OPEN_EXR'
    fileOutput.format.color_depth = '32'
    fileOutput.format.color_management = 'OVERRIDE'
    tree.links.new(rl.outputs[2], fileOutput.inputs[0])
    render_scene("/tmp/scene.png", False)
    depth_fn = os.path.join(cur_base_path, f"{img_id:05}_depth.exr")
    exr_path = os.path.join("/tmp/depth", os.listdir("/tmp/depth")[0])
    img = np.copy(read_exr_single_channel(exr_path))
    unique_vals = sorted(np.unique(img))
    nmax = unique_vals[-2]
    nmin = np.min(img)
    img[img == unique_vals[-1]] = nmax
    img = (img - nmin) / (nmax - nmin)
    save_exr_single_channel(exr_path, img)
    shutil.move(exr_path, depth_fn)
    fileOutput.mute = True
    return depth_fn

def render_shading_image(cur_base_path, img_id, ply_obj_name, config_yaml):
    """
    Render the emissive image and the white (shadow catcher) image to compute shading without shadows.

    Args:
        cur_base_path (str): Base directory for outputs.
        img_id (int): Image ID.
        ply_obj_name (str): Name of the PLY object.
        config_yaml (dict): Configuration dictionary.

    Returns:
        tuple: (Path to shading image, shading image as NumPy array)
    """
    ply_obj = bpy.data.objects.get(ply_obj_name)
    material = bpy.data.materials.new(name="Default_Material")
    material.use_nodes = True
    if len(ply_obj.data.materials) == 0:
        ply_obj.data.materials.append(material)
    else:
        ply_obj.data.materials[0] = material
    ply_obj.is_shadow_catcher = True
    bpy.context.scene.view_layers["ViewLayer"].use_pass_z = False
    bpy.context.scene.view_layers["ViewLayer"].cycles.use_pass_shadow_catcher = True
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    for n in tree.nodes:
        tree.nodes.remove(n)
    rl = tree.nodes.new('CompositorNodeRLayers')
    fileOutput = tree.nodes.new(type="CompositorNodeOutputFile")
    fileOutput.base_path = "/tmp/white"
    fileOutput.format.file_format = 'OPEN_EXR'
    fileOutput.format.color_depth = '32'
    fileOutput.format.color_management = 'OVERRIDE'
    fileOutput.format.color_mode = 'RGBA'
    tree.links.new(rl.outputs["Shadow Catcher"], fileOutput.inputs[0])
    render_scene("/tmp/scene_white.png", False)
    ply_obj.is_shadow_catcher = False
    bpy.context.scene.view_layers["ViewLayer"].cycles.use_pass_shadow_catcher = False
    emissive_image = cv2.imread("/tmp/scene.png")
    white_exr = os.path.join("/tmp/white", os.listdir("/tmp/white")[0])
    shadow_catcher_image = read_exr_single_channel(white_exr, 'B')
    shadow_catcher_image = np.where(
        shadow_catcher_image <= 0.0031308,
        12.92 * shadow_catcher_image,
        1.055 * np.power(shadow_catcher_image, 1/2.4) - 0.055
    )
    inverted_image = 1 - shadow_catcher_image
    black_image = np.zeros((inverted_image.shape[0], inverted_image.shape[1], 3), dtype=np.float32)
    emissive_image = emissive_image.astype(np.float32) / 255.0
    alpha = inverted_image
    alpha_rgb = np.stack((alpha, alpha, alpha), axis=-1)
    foreground = cv2.multiply(alpha_rgb, black_image)
    background = cv2.multiply(1.0 - alpha_rgb, emissive_image)
    scene_shading = cv2.add(foreground, background)
    shading_fn = os.path.join(cur_base_path, f"{img_id:05}.png")
    cv2.imwrite(shading_fn, scene_shading * 255, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    fileOutput.mute = True
    return shading_fn, scene_shading

def render_normal_map(cur_base_path, img_id):
    """
    Render the normal map of the scene and save it as an EXR file.

    Args:
        cur_base_path (str): Base directory for outputs.
        img_id (int): Image ID.

    Returns:
        str: Path to the saved normal map.
    """
    bpy.context.scene.view_layers["ViewLayer"].use_pass_normal = True
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    for n in tree.nodes:
        tree.nodes.remove(n)
    rl = tree.nodes.new('CompositorNodeRLayers')
    fileOutput = tree.nodes.new(type="CompositorNodeOutputFile")
    fileOutput.base_path = "/tmp/normal"
    fileOutput.format.file_format = 'OPEN_EXR'
    fileOutput.format.color_depth = '32'
    fileOutput.format.color_management = 'OVERRIDE'
    tree.links.new(rl.outputs[2], fileOutput.inputs[0])
    render_scene("/tmp/scene_white_.png", False)
    normal_fn = os.path.join(cur_base_path, f"{img_id:05}_normal.exr")
    normal_path = os.path.join("/tmp/normal", os.listdir("/tmp/normal")[0])
    shutil.move(normal_path, normal_fn)
    fileOutput.mute = True
    bpy.context.scene.view_layers["ViewLayer"].use_pass_normal = False
    return normal_fn

def render_masks(cur_base_path, img_id, cubes, type_index_dict):
    """
    Render class and instance masks using object index passes and save them as EXR files.

    Args:
        cur_base_path (str): Base directory for outputs.
        img_id (int): Image ID.
        cubes (list): List of mesh objects.
        type_index_dict (dict): Mapping from BoxType to class index.

    Returns:
        tuple: (Path to class mask, Path to instance mask)
    """
    bpy.context.scene.view_layers["ViewLayer"].use_pass_object_index = True
    tree = bpy.context.scene.node_tree
    fileOutput = tree.nodes.new(type="CompositorNodeOutputFile")
    fileOutput.base_path = "/tmp/classes_mask"
    fileOutput.format.file_format = 'OPEN_EXR'
    fileOutput.format.color_depth = '32'
    fileOutput.format.color_management = 'OVERRIDE'
    tree.links.new(tree.nodes["Render Layers"].outputs["IndexOB"], fileOutput.inputs[0])
    for cube in cubes:
        if cube.type == 'MESH':
            cube.pass_index = type_index_dict[cube["BoxType"]]
    render_scene("/tmp/classes.png", True)
    classes_fn = os.path.join(cur_base_path, f"{img_id:05}_classes.exr")
    classes_path = os.path.join("/tmp/classes_mask", os.listdir("/tmp/classes_mask")[0])
    shutil.move(classes_path, classes_fn)
    fileOutput.mute = True
    fileOutput = tree.nodes.new(type="CompositorNodeOutputFile")
    fileOutput.base_path = "/tmp/instances_mask"
    fileOutput.format.file_format = 'OPEN_EXR'
    fileOutput.format.color_depth = '32'
    fileOutput.format.color_management = 'OVERRIDE'
    tree.links.new(tree.nodes["Render Layers"].outputs["IndexOB"], fileOutput.inputs[0])
    for idx, cube in enumerate(cubes, start=1):
        if cube.type == 'MESH':
            cube.pass_index = idx
    render_scene("/tmp/instances.png", True)
    instances_fn = os.path.join(cur_base_path, f"{img_id:05}_instances.exr")
    instances_path = os.path.join("/tmp/instances_mask", os.listdir("/tmp/instances_mask")[0])
    shutil.move(instances_path, instances_fn)
    fileOutput.mute = True
    bpy.context.scene.view_layers["ViewLayer"].use_pass_object_index = False
    return classes_fn, instances_fn

def render_3d_bboxes(cur_base_path, img_id, scene_shading_none, data_json, ply_obj_name):
    """
    Project 3D bounding boxes of scene objects (excluding the PLY object, camera, and lights)
    onto the image, draw lines and points, and update the JSON data with object information.

    Args:
        cur_base_path (str): Base directory for outputs.
        img_id (int): Image ID.
        scene_shading_none (ndarray): Unmodified shading image as a NumPy array.
        data_json (dict): JSON structure to update with object info.
        ply_obj_name (str): Name of the PLY object to ignore.

    Returns:
        str: Path to the saved image with rendered 3D bounding boxes.
    """
    bbox3d_image = scene_shading_none * 255
    connections = [
        (0, 1), (0, 3), (0, 4),
        (1, 2), (2, 3), (4, 7),
        (4, 5), (6, 7), (5, 6),
        (1, 5), (2, 6), (3, 7)
    ]
    scene = bpy.context.scene
    camera = scene.camera
    resolution_x = scene.render.resolution_x
    resolution_y = scene.render.resolution_y
    sensor_width = camera.data.sensor_width
    sensor_height = camera.data.sensor_height
    focal_length_mm = camera.data.lens
    fx = (focal_length_mm * resolution_x) / sensor_width
    fy = (focal_length_mm * resolution_y) / sensor_height
    cx = resolution_x / 2
    cy = resolution_y / 2

    light_prefixes = ["Point", "Sun", "Area", "Spot"]
    for obj in bpy.context.scene.objects:
        if obj.name == ply_obj_name or obj.name == "Camera" or any(obj.name.startswith(prefix) for prefix in light_prefixes):
            continue
        if "BoxType" not in obj:
            continue
        bbox = [Vector(corner) for corner in obj.bound_box]
        matrix_world = obj.matrix_world
        screen_positions = [world_to_camera_view_own(scene, obj, corner, camera, matrix_world) for corner in bbox]
        points_px = [(int(pos[0]), int(pos[1])) for pos in screen_positions]
        translation, rotation = get_rotation_translation(obj)
        add_object_to_image(f"{img_id:05}", obj.name, obj["BoxType"], rotation, translation, (fx, fy), (cx, cy), points_px, data_json)
        for start_idx, end_idx in connections:
            start = points_px[start_idx]
            end = points_px[end_idx]
            cv2.line(bbox3d_image, (start[0], start[1]), (end[0], end[1]), color=(0, 0, 255), thickness=2)
        for (x, y) in points_px:
            cv2.circle(bbox3d_image, (x, y), radius=3, color=(0, 255, 0), thickness=-1)
    bbox3d_fn = os.path.join(cur_base_path, f"{img_id:05}_boxes3d.png")
    cv2.imwrite(bbox3d_fn, bbox3d_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    return bbox3d_fn

