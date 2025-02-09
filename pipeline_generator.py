import csv
import json
import os
import random
import time
import cv2
import bpy

from tabulate import tabulate

from init_scene import init_scene, load_scene_and_init_camera
from lighting import init_lights
from objects import random_rects, create_scene_objects, remove_invisible_objects, remove_intersections, remove_object_intersections
from camera import adjust_camera
from rendering import render_depth_map, render_shading_image, render_normal_map, render_masks, render_3d_bboxes
from utils import clean_objects_from_scene, clean_blender_keep_ply, compute_bounding_boxes

# =============================================================================
# STEP 1: Initialization and Scene Creation
# =============================================================================

def step1_init_scene(assets_obj, obj_files, dict_prisms, dict_assets, config_yaml, ply_obj_name, obj_ply_BVHtree):
    """
    Performs initialization and scene creation.

    - Initializes the statistics dictionaries.
    - Selects a random object file and determines its type.
    - Chooses random prism/asset types and computes dimensions.
    - Calls init_scene and load_scene_and_init_camera.
    - Initializes lights.
    - Creates scene objects (with "BoxType" assigned) via create_scene_objects.
    - Adjusts the camera and removes invisible or intersecting objects.

    Returns:
        cubes: List of created scene objects.
        stats_boxes_aux: Dictionary with per-class counts.
        stats_boxes_rej_aux: Dictionary with per-class rejected counts.
        focal_length: Focal length of the camera.
        lights_info: Information about the lights.

    If no objects remain (due to removals), returns None.
    """
    # Initialize statistics dictionaries
    stats_boxes_aux = {}
    stats_boxes_rej_aux = {}

    for key in config_yaml["prism_types"] + assets_obj:
        if key.endswith('.obj'):
            stats_boxes_aux[key[:-4]], stats_boxes_rej_aux[key[:-4]] = 0, 0
        else:
            stats_boxes_aux[key], stats_boxes_rej_aux[key] = 0, 0

    # Determine file type (obj, blend or ply)
    obj_f = blend_f = ply_f = False
    obj_file = random.choice(obj_files)
    if obj_file.endswith(".obj"):
        obj_f = True
    elif obj_file.endswith(".blend"):
        blend_f = True
    elif obj_file.endswith(".ply"):
        ply_f = True

    # Choose random prism/asset types
    rnd_prisms_types = random.choices(config_yaml["prism_types"] + assets_obj, k=config_yaml["num_instances"])
    assets_files = [file[:-4] for file in rnd_prisms_types if file.endswith('.obj')]
    non_obj_files = [file for file in rnd_prisms_types if not file.endswith('.obj')]

    # Compute dimensions
    base_dims_prisms = [(dict_prisms[rp][0], dict_prisms[rp][2]) for rp in non_obj_files]
    heigth_dim_prisms = [dict_prisms[rp][1] for rp in non_obj_files]
    base_dims_assets = [(dict_assets[ra][0], dict_assets[ra][2]) for ra in assets_files]
    heigth_dim_assets = [dict_assets[ra][1] for ra in assets_files]

    # Initialize scene and load ply object
    init_scene(ply_obj_name)
    scene_obj, random_index, camera_pos, camera_rot, focal_length = load_scene_and_init_camera(
        obj_file, config_yaml["directory"], obj_f, blend_f, ply_f, ply_obj_name,
        config_yaml["camera_position"], config_yaml["camera_rotation_constraints"],
        config_yaml["focal_length_constraints"]
    )
    lights_info = init_lights(config_yaml["lightning_constraints"])

    # Generate random parameters for prisms/assets and create scene objects.
    rnd_prisms = list(zip(non_obj_files + assets_files,
                          random_rects(base_dims_prisms + base_dims_assets,
                                       heigth_dim_prisms + heigth_dim_assets,
                                       config_yaml["maxtries"], scene_obj, len(base_dims_assets)),
                          heigth_dim_prisms + heigth_dim_assets))
    cubes = create_scene_objects(rnd_prisms, assets_files, scene_obj, config_yaml)

    # Adjust camera and filter objects
    adjust_camera(cubes)
    cubes = remove_invisible_objects(cubes)
    if len(cubes) == 0:
        clean_blender_keep_ply(ply_obj_name)
        return None
    cubes = remove_intersections(cubes, obj_ply_BVHtree)
    if len(cubes) == 0:
        clean_blender_keep_ply(ply_obj_name)
        return None
    cubes = remove_object_intersections(cubes)
    if len(cubes) == 0:
        clean_blender_keep_ply(ply_obj_name)
        return None

    return cubes, stats_boxes_aux, stats_boxes_rej_aux, focal_length, lights_info

# =============================================================================
# STEP 2: Render Maps
# =============================================================================

def step2_render_maps(cur_base_path, img_id, ply_obj_name, config_yaml, data_json, cubes,
                      type_index_dict, boxes_types, stats_boxes_aux, stats_boxes_rej_aux):
    """
    Renders all required maps:
      - Depth map.
      - Shading image (emissive and white) to obtain scene_shading_none.
      - Normal map.
      - 3D bounding boxes (processing only the cubes list, which guarantees "BoxType").
      - Class and instance masks.
      - Computes 2D bounding boxes from the instance mask.

    Returns:
        scene_shading_none, bbxs, bbxs_rej, cubes_rejected, cubes_remaining.
    """
    depth_fn = render_depth_map(cur_base_path, img_id)
    shading_fn, scene_shading_none = render_shading_image(cur_base_path, img_id, ply_obj_name, config_yaml)
    normal_fn = render_normal_map(cur_base_path, img_id)
    bbox3d_fn = render_3d_bboxes(cur_base_path, img_id, scene_shading_none, data_json, cubes)
    classes_fn, instances_fn = render_masks(cur_base_path, img_id, cubes, type_index_dict)
    bbxs, bbxs_rej, cubes_rejected, cubes_remaining = compute_bounding_boxes(
        instances_fn, classes_fn, scene_shading_none, cubes, config_yaml,
        boxes_types, data_json, img_id, stats_boxes_aux, stats_boxes_rej_aux)
    return scene_shading_none, bbxs, bbxs_rej, cubes_rejected, cubes_remaining

# =============================================================================
# STEP 3: Validate Scene
# =============================================================================

def step3_validate_scene(bbxs, bbxs_rej, config_yaml, data_json, img_id,
                         ply_obj_name, cubes_rejected, cubes_remaining):
    """
    Validates the scene based on the number of accepted instances.

    - If the number of accepted bounding boxes (bbxs) is below the minimum threshold, the scene is rejected.
    - If there are rejected objects, the scene is modified by removing them.
    - If the scene meets the criteria, it is accepted.

    Returns:
      (valid_flag, new_cubes)
      valid_flag can be:
        - False: Scene rejected
        - "modified": Scene modified (rejected objects removed)
        - True: Scene accepted
      new_cubes contains the updated list of objects if applicable.
    """
    img_key = f"{img_id:05}"

    # Reject the scene if the number of accepted objects is below the required threshold
    if len(bbxs) < config_yaml["min_instances_accepted"]:
        # Remove the last generated image from JSON (if it exists)
        try:
            if img_key in data_json["images"]:
                data_json["images"].pop(img_key)
        except:
            pass  # Prevent errors if the key does not exist

        # Clean the scene while keeping the PLY object
        clean_blender_keep_ply(ply_obj_name)

        return False, None  # Scene is rejected

    # If there are rejected objects, modify the scene by removing them
    if len(bbxs_rej) > 0:
        clean_objects_from_scene(cubes_rejected)  # Remove rejected objects

        # Update object list after removing rejected ones
        cubes = cubes_remaining

        # Apply material to the remaining objects
        material_vertexcolor = bpy.data.materials.get("VertexColorMaterial")
        ply_obj = bpy.data.objects.get(ply_obj_name)
        ply_obj.data.materials[0] = material_vertexcolor

        # Remove the image entry from JSON (if it exists)
        try:
            if img_key in data_json["images"]:
                data_json["images"].pop(img_key)
        except:
            pass  # Prevent errors if the key does not exist

        return "modified", cubes  # Scene is modified but not rejected

    return True, None  # Scene is accepted

# =============================================================================
# STEP 4: Write Outputs and Cleanup
# =============================================================================

def step4_write_outputs(cur_base_path, img_id, bbxs, data_json, stats_boxes_ok_scene_gen, stats_boxes_rej_scene_gen,
                        stats_boxes_aux, stats_boxes_rej_aux, config_yaml, scene_shading_none,
                        lights_info, start_time, modified_scenes_before_accept, rejected_scenes_before_accept,
                        focal_length, csv_glbl_fn, write_glbl_csv_header, json_path):
    """
    Writes output files (KITTI txt, CSV, bounding boxes image) and updates the report.
    The report_info.txt file is saved in the parent directory of cur_base_path (i.e., output_dir).
    Returns the updated write_glbl_csv_header flag.
    """
    # Update the JSON (if necessary).
    with open(json_path, 'w') as f:
        json.dump(data_json, f, indent=4)

    # Update global statistics with the accepted instances
    stats_boxes_ok_scene_gen.update({key: stats_boxes_ok_scene_gen[key] + stats_boxes_aux[key] for key in stats_boxes_ok_scene_gen})
    stats_boxes_rej_scene_gen.update({key: stats_boxes_rej_scene_gen[key] + stats_boxes_rej_aux[key] for key in stats_boxes_rej_scene_gen})

    # Write KITTI txt file using the bbxs list (which contains the expected keys)
    txt_fn = os.path.join(cur_base_path, f"{img_id:05}.txt")
    with open(txt_fn, 'w') as f:
        for bbx in bbxs:
            f.write(f"{bbx['cls_name']} 0.0 0 0.0 {bbx['x_min']} {bbx['y_min']} {bbx['x_max']} {bbx['y_max']} 0.0 0.0 0.0 0.0 0.0 0.0 0.0\n")

    # Draw and save boxes image
    box_image = scene_shading_none * 255
    for bbx in bbxs:
        cv2.rectangle(box_image, (bbx["x_min"], bbx["y_min"]),
                      (bbx["x_max"], bbx["y_max"]), (0, 0, 255), 2)
        cv2.putText(box_image, f"{bbx['cls_name']}", (bbx["x_min"], bbx["y_max"] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(box_image, f"(cvr={bbx['cvr']}%)", (bbx["x_min"], bbx["y_min"] + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    box_fn = os.path.join(cur_base_path, f"{img_id:05}_boxes.png")
    cv2.imwrite(box_fn, box_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    # Write CSV file using the bbxs list
    csv_fn = os.path.join(cur_base_path, f"{img_id:05}.csv")
    with open(csv_fn, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=bbxs[0].keys())
        writer.writeheader()
        writer.writerows(bbxs)

    # Prepare report using tabulate
    end_time = time.time()
    elapsed = end_time - start_time
    camera = bpy.context.scene.camera
    pos = (round(camera.location.x, 2), round(camera.location.y, 2), round(camera.location.z, 2))
    rot = (round(camera.rotation_euler.x, 2), round(camera.rotation_euler.y, 2), round(camera.rotation_euler.z, 2))
    camera_row = f"Camera position: {pos} | Camera rotation: {rot} | Focal length: {focal_length}"
    time_row = f"Scene generation time: {elapsed:.2f} seconds | Scenes modified: {modified_scenes_before_accept} | Scenes rejected: {rejected_scenes_before_accept}"
    table = tabulate([list(bbx.values()) for bbx in bbxs],
                     headers=bbxs[0].keys(), tablefmt='fancy_grid')
    table += "\n" + tabulate([[camera_row]], tablefmt='fancy_grid', colalign=["left"])
    table += "\n" + tabulate([[time_row]], tablefmt='fancy_grid', colalign=["left"])

    # Save report_info.txt in the parent directory of cur_base_path (output_dir)
    rep_fn = os.path.join(os.path.dirname(cur_base_path), "report_info.txt")
    with open(rep_fn, 'a', newline='') as rep_file:
        rep_file.write(table)
        rep_file.write("\n" + "-" * 110 + "\n")

    # Write global CSV info
    with open(csv_glbl_fn, 'a', newline='') as csv_glbl_file:
        writer = csv.DictWriter(csv_glbl_file, fieldnames=bbxs[0].keys())
        if write_glbl_csv_header:
            writer.writeheader()
            write_glbl_csv_header = False
        writer.writerows(bbxs)

    return write_glbl_csv_header

# =============================================================================
# MAIN FUNCTION: generate_scene
# =============================================================================

def generate_scene(assets_obj, obj_files, dict_prisms, dict_assets, cur_base_path, data_json,
                   type_index_dict, boxes_types, stats_boxes_ok_scene_rej, stats_boxes_rej_scene_rej,
                   stats_boxes_ok_scene_mod, stats_boxes_rej_scene_mod,
                   stats_boxes_ok_scene_gen, stats_boxes_rej_scene_gen, csv_glbl_fn,
                   tot_scene_rejected, tot_scene_modified, tot_scene_rej_with_rej,
                   tot_scene_rej_without_rej, write_glbl_csv_header, ply_obj_name, config_yaml,
                   img_id, obj_ply_BVHtree, json_path):
    """
    Main function that generates a scene.

    It iterates until a scene is accepted, following these steps:
      1. Initialization and scene creation.
      2. Render maps.
      3. Scene validation.
      4. Write outputs and cleanup.
    """
    start_time = time.time()
    generate_again = True
    generate_again_deleting_objects = False
    rejected_scenes_before_accept = 0
    modified_scenes_before_accept = 0

    while generate_again:
        # --- Step 1: Initialization and Scene Creation ---
        if not generate_again_deleting_objects:
            result = step1_init_scene(assets_obj, obj_files, dict_prisms, dict_assets, config_yaml, ply_obj_name, obj_ply_BVHtree)
            if result is None:
                # No objects remain; clean and try again
                clean_blender_keep_ply(ply_obj_name)
                generate_again = True
                continue
            cubes, stats_boxes_aux, stats_boxes_rej_aux, focal_length, lights_info = result

        # --- Step 2: Render Maps ---
        scene_shading_none, bbxs, bbxs_rej, cubes_rejected, cubes_remaining = step2_render_maps(
            cur_base_path, img_id, ply_obj_name, config_yaml, data_json, cubes,
            type_index_dict, boxes_types, stats_boxes_aux, stats_boxes_rej_aux)

        # --- Step 3: Validate Scene ---
        valid_flag, new_cubes = step3_validate_scene(bbxs, bbxs_rej, config_yaml, data_json, img_id,
                                                      ply_obj_name, cubes_rejected, cubes_remaining)

        if valid_flag is False:
            generate_again_deleting_objects = False
            generate_again = True
            tot_scene_rejected = tot_scene_rejected + 1
            rejected_scenes_before_accept += 1  # Increase count of rejected scenes before acceptance

            # Track rejected scenes with and without rejected objects
            if len(bbxs_rej) > 0:
                tot_scene_rej_with_rej += 1
            else:
                tot_scene_rej_without_rej += 1

            # Update statistics for accepted/rejected instances in REJECTED scenes
            stats_boxes_ok_scene_rej.update({key: stats_boxes_ok_scene_rej[key] + stats_boxes_aux[key] for key in stats_boxes_ok_scene_rej})
            stats_boxes_rej_scene_rej.update({key: stats_boxes_rej_scene_rej[key] + stats_boxes_rej_aux[key] for key in stats_boxes_rej_scene_rej})

            continue

        if valid_flag == "modified":
            generate_again_deleting_objects = True
            generate_again = True
            tot_scene_modified = tot_scene_modified + 1
            modified_scenes_before_accept += 1  # Increase count of modified scenes

            # Update statistics for accepted/rejected instances in MODIFIED scenes
            stats_boxes_ok_scene_mod.update({key: stats_boxes_ok_scene_mod[key] + stats_boxes_aux[key] for key in stats_boxes_ok_scene_mod})
            stats_boxes_rej_scene_mod.update({key: stats_boxes_rej_scene_mod[key] + stats_boxes_rej_aux[key] for key in stats_boxes_rej_scene_mod})

            # Reset stats_boxes_aux to avoid accumulating values multiple times when the scene is finally accepted
            for key in stats_boxes_aux:
                stats_boxes_aux[key] = 0

            cubes = new_cubes
            continue

        generate_again = False # Scene accepted

        # --- Step 4: Write Outputs and Cleanup ---
        with open(json_path, 'w') as f:
            json.dump(data_json, f, indent=4)

        # Use the bbxs list computed in Step 2 to write outputs (avoiding KeyError)
        write_glbl_csv_header = step4_write_outputs(cur_base_path, img_id, bbxs, data_json,
                                                     stats_boxes_ok_scene_gen, stats_boxes_rej_scene_gen,
                                                     stats_boxes_aux, stats_boxes_rej_aux, config_yaml,
                                                     scene_shading_none, lights_info, start_time,
                                                     modified_scenes_before_accept, rejected_scenes_before_accept,
                                                     focal_length, csv_glbl_fn, write_glbl_csv_header, json_path)

        clean_blender_keep_ply(ply_obj_name)

    return tot_scene_rejected, tot_scene_modified, tot_scene_rej_with_rej, tot_scene_rej_without_rej, write_glbl_csv_header
