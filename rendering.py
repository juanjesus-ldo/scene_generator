import os
import bpy
import numpy as np
import shutil
import cv2
import mathutils

from utils import (
    stdout_redirected, stderr_redirected, read_exr_single_channel, save_exr_single_channel,
    world_to_camera_view_own, get_rotation_translation, add_object_to_image
)

def render_scene(outfile, bbg):
    """
    Render the current scene to the specified output file.

    If bbg is True, the background color is set to black.

    Args:
        outfile (str): Path to the output file.
        bbg (bool): If True, set the background color to black.
    """
    if bbg:
        # Set background color as black
        if bpy.context.scene.world is None:
            bpy.context.scene.world = bpy.data.worlds.new("World")
        bpy.context.scene.world.use_nodes = True
        bg = bpy.context.scene.world.node_tree.nodes['Background']
        bg.inputs[0].default_value = (0, 0, 0, 1)  # RGB + Alpha
    bpy.context.scene.render.filepath = outfile
    with stdout_redirected():
        bpy.ops.render.render(write_still=True)


def render_depth_map(cur_base_path, img_id):
    """
    Render the depth map, process and normalize it, and save it as an EXR file.

    Args:
        cur_base_path (str): Base directory for output files.
        img_id (int): Image identifier used for naming the output file.

    Returns:
        str: File path to the saved depth map.
    """
    bpy.context.scene.view_layers["ViewLayer"].use_pass_z = True
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    # Remove all nodes from the node tree
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
    Render the emissive image and the white image to compute shading without shadows.

    Sets up a default material on the PLY object, enables shadow catcher mode, renders the scene,
    and then combines the images to compute the final shading.

    Args:
        cur_base_path (str): Base directory for output files.
        img_id (int): Image identifier used for naming the output file.
        ply_obj_name (str): Name of the PLY object.
        config_yaml (dict): Configuration dictionary.

    Returns:
        tuple: (File path to the saved shading image, shading image as a NumPy array)
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
    Render the normal map of the current scene and save it as an EXR file.

    Args:
        cur_base_path (str): Base directory for output files.
        img_id (int): Image identifier used for naming the output file.

    Returns:
        str: File path to the saved normal map.
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
    Render class and instance masks for the scene.

    Generates two EXR files: one for the class mask (using the BoxType property mapped via type_index_dict)
    and one for the instance mask (with unique indices for each object).

    Args:
        cur_base_path (str): Base directory for output files.
        img_id (int): Image identifier used for naming the output file.
        cubes (list): List of mesh objects.
        type_index_dict (dict): Dictionary mapping BoxType to class index.

    Returns:
        tuple: (File path to class mask, File path to instance mask)
    """
    bpy.context.scene.view_layers["ViewLayer"].use_pass_object_index = True
    tree = bpy.context.scene.node_tree

    # Render class mask
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

    # Render instance mask
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
    Render 3D bounding boxes on the shading image and update a JSON structure with object data.

    Projects the 3D bounding box corners of objects (excluding the PLY object, Camera, and lights)
    to screen space, draws lines and points for the bounding boxes on the image,
    and calls a helper function to add object information to data_json.

    Args:
        cur_base_path (str): Base directory for output files.
        img_id (int): Image identifier used for naming the output file.
        scene_shading_none (ndarray): Shading image (without modifications) as a NumPy array.
        data_json (dict): JSON structure for storing object data.
        ply_obj_name (str): Name of the PLY object to ignore.

    Returns:
        str: File path to the image with rendered 3D bounding boxes.
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

    # Define prefixes of objects to ignore (Camera, PLY object, and lights)
    light_prefixes = ["Point", "Sun", "Area", "Spot"]
    for obj in bpy.context.scene.objects:
        if obj.name == ply_obj_name or obj.name == "Camera" or any(obj.name.startswith(prefix) for prefix in light_prefixes):
            continue
        if "BoxType" not in obj:
            continue
        bbox = [mathutils.Vector(corner) for corner in obj.bound_box]
        matrix_world = obj.matrix_world
        screen_positions = [
            world_to_camera_view_own(scene, obj, corner, camera, matrix_world) for corner in bbox
        ]
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
