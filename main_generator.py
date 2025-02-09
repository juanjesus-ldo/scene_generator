#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main generator script.
Usage: time ./main_generator.py --config_file_path ./config.yaml
"""

import argparse
import bpy
import random
import os
import shutil
import sys
import bmesh
import time
import json
import signal
import re
import io
import contextlib

from mathutils.bvhtree import BVHTree
from datetime import datetime

from config import load_config, validate_exclusive_args, validate_min_instances, validate_camera_positions, validate_rotation_constraints, validate_focal_length_constraints, validate_lightning_constraints
from utils import get_obj_dimensions, set_render_dimensions, initialize_stats, generate_unique_color
from pipeline_generator import generate_scene

# Global variables for statistics and reporting
start_time_main = 0
total_scenes_global = None
tot_scene_accepted_global = 0
tot_scene_rejected_global = None
tot_scene_modified_global = None
tot_scene_rej_with_rej_global = None
tot_scene_rej_without_rej_global = None
total_exec_time_global = None
avg_time_all_global = None
avg_time_accepted_global = None
report_file_path_previous_exists = False
rep_fn = None
stats_written = False  # Global flag to prevent duplicate execution
stats_boxes_ok_scene_gen_global = {}
stats_boxes_rej_scene_gen_global = {}
stats_boxes_ok_scene_mod_global = {}
stats_boxes_rej_scene_mod_global = {}
stats_boxes_ok_scene_rej_global = {}
stats_boxes_rej_scene_rej_global = {}

def setup_output_directory(output_dir):
    """
    Setup the output directory. If the directory name ends with "-r", the base directory is removed.
    Returns the base output directory and a reset flag.
    """
    reset_flag = output_dir.endswith("-r")
    base_output_dir = output_dir[:-2] if reset_flag else output_dir
    if reset_flag:
        if os.path.isdir(base_output_dir):
            shutil.rmtree(base_output_dir)
        os.makedirs(base_output_dir, exist_ok=True)
    else:
        os.makedirs(base_output_dir, exist_ok=True)
    return base_output_dir, reset_flag

def load_previous_stats(report_file_path):
    """
    Load previous statistics from the report file and update global variables.
    """
    global total_exec_time_global, avg_time_all_global, avg_time_accepted_global
    global tot_scene_rej_with_rej_global, tot_scene_rej_without_rej_global
    global stats_boxes_rej_scene_rej_global, stats_boxes_ok_scene_rej_global
    global stats_boxes_rej_scene_mod_global, tot_scene_modified_global, stats_boxes_ok_scene_mod_global
    global stats_boxes_ok_scene_gen_global, total_scenes_global, tot_scene_accepted_global, tot_scene_rejected_global

    with open(report_file_path, 'r') as report_file:
        for line in report_file.readlines()[::-1]:
            m = re.search(r"Total execution time since start: ([\d.]+) hours", line)
            if m:
                total_exec_time_global = float(m.group(1))
                continue
            m = re.search(r"Average scene generation time \(accepted, modified, and rejected scenes - (\d+)\): ([\d.]+) seconds", line)
            if m:
                avg_time_all_global = float(m.group(2))
                continue
            m = re.search(r"Average scene generation time \(accepted scenes only - (\d+)\): ([\d.]+) seconds", line)
            if m:
                avg_time_accepted_global = float(m.group(2))
                continue
            m = re.search(r"Total instances rejected in (\d+) images rejected \(with \[(\d+)\] / without \[(\d+)\] rejected instances\): (\d+) \| (.+)", line)
            if m:
                tot_scene_rej_with_rej_global = int(m.group(2))
                tot_scene_rej_without_rej_global = int(m.group(3))
                stats_boxes_rej_scene_rej_global = eval(m.group(5))
                continue
            m = re.search(r"Total instances accepted in (\d+) images rejected: (\d+) \| (.+)", line)
            if m:
                stats_boxes_ok_scene_rej_global = eval(m.group(3))
                continue
            m = re.search(r"Total instances rejected in (\d+) images modified: (\d+) \| (.+)", line)
            if m:
                stats_boxes_rej_scene_mod_global = eval(m.group(3))
                continue
            m = re.search(r"Total instances accepted in (\d+) images modified: (\d+) \| (.+)", line)
            if m:
                tot_scene_modified_global = int(m.group(1))
                stats_boxes_ok_scene_mod_global = eval(m.group(3))
                continue
            m = re.search(r"Total instances accepted in (\d+) images generated: (\d+) \| (.+)", line)
            if m:
                stats_boxes_ok_scene_gen_global = eval(m.group(3))
                continue
            m = re.search(r"Total scenes attempted: (\d+) \((\d+) accepted \| (\d+) modified \| (\d+) rejected\)", line)
            if m:
                total_scenes_global = int(m.group(1))
                tot_scene_accepted_global = int(m.group(2))
                tot_scene_modified_global = int(m.group(3))
                tot_scene_rejected_global = int(m.group(4))
                break

def initialize_stats(prism_types, assets_obj):
    """
    Initialize bounding box statistics dictionaries for each object type.
    Returns dictionaries for accepted and rejected boxes for generated, modified, and rejected scenes.
    """
    global stats_boxes_ok_scene_rej;  stats_boxes_ok_scene_rej = {}
    global stats_boxes_rej_scene_rej; stats_boxes_rej_scene_rej = {}
    global stats_boxes_ok_scene_mod; stats_boxes_ok_scene_mod = {}
    global stats_boxes_rej_scene_mod; stats_boxes_rej_scene_mod = {}
    global stats_boxes_ok_scene_gen; stats_boxes_ok_scene_gen = {}
    global stats_boxes_rej_scene_gen; stats_boxes_rej_scene_gen = {}

    for key in prism_types + assets_obj:
        label = key[:-4] if key.endswith('.obj') else key
        stats_boxes_ok_scene_rej[label] = 0
        stats_boxes_rej_scene_rej[label] = 0
        stats_boxes_ok_scene_mod[label] = 0
        stats_boxes_rej_scene_mod[label] = 0
        stats_boxes_ok_scene_gen[label] = 0
        stats_boxes_rej_scene_gen[label] = 0

    return (stats_boxes_ok_scene_rej, stats_boxes_rej_scene_rej,
            stats_boxes_ok_scene_mod, stats_boxes_rej_scene_mod,
            stats_boxes_ok_scene_gen, stats_boxes_rej_scene_gen)

def write_stats_signal_handler(sig, frame):
    """
    Signal handler that writes job statistics to a report file and prints them to the console.

    This function aggregates statistics for scenes generated (accepted, modified, rejected),
    including instance counts for different categories, calculates average generation times,
    and writes the results to both a report file and the console before exiting the program.

    Args:
        sig: Signal number.
        frame: Current stack frame.
    """
    global start_time_main, total_scenes_global, tot_scene_accepted_global, tot_scene_rejected_global, tot_scene_modified_global
    global tot_scene_rej_with_rej_global, tot_scene_rej_without_rej_global
    global total_exec_time_global, avg_time_all_global, avg_time_accepted_global
    global report_file_path_previous_exists, rep_fn, stats_written
    global stats_boxes_ok_scene_gen_global, stats_boxes_rej_scene_gen_global
    global stats_boxes_ok_scene_mod_global, stats_boxes_rej_scene_mod_global
    global stats_boxes_ok_scene_rej_global, stats_boxes_rej_scene_rej_global
    global stats_boxes_ok_scene_gen, stats_boxes_rej_scene_gen
    global stats_boxes_ok_scene_mod, stats_boxes_rej_scene_mod
    global stats_boxes_ok_scene_rej, stats_boxes_rej_scene_rej

    if stats_written:
        return  # If stats have already been written, do nothing

    stats_written = True
    # Build statistics report lines for bounding box stats (printed and saved)
    lines = ["\n"]
    lines.append("FINISHED JOB STATS:\n\n")
    lines.append(f"Total scenes attempted: {tot_scene_accepted + tot_scene_rejected + tot_scene_modified} "
                 f"({tot_scene_accepted} accepted | {tot_scene_modified} modified | {tot_scene_rejected} rejected)\n")
    lines.append(f"\nTotal instances accepted in {tot_scene_accepted} images generated: {sum(stats_boxes_ok_scene_gen.values())} | {stats_boxes_ok_scene_gen}\n\n")
    lines.append(f"Total instances accepted in {tot_scene_modified} images modified: {sum(stats_boxes_ok_scene_mod.values())} | {stats_boxes_ok_scene_mod}\n")
    lines.append(f"Total instances rejected in {tot_scene_modified} images modified: {sum(stats_boxes_rej_scene_mod.values())} | {stats_boxes_rej_scene_mod}\n\n")
    lines.append(f"Total instances accepted in {tot_scene_rejected} images rejected: {sum(stats_boxes_ok_scene_rej.values())} | {stats_boxes_ok_scene_rej}\n")
    lines.append(f"Total instances rejected in {tot_scene_rejected} images rejected (with [{tot_scene_rej_with_rej}] / without [{tot_scene_rej_without_rej}] rejected instances): "
                 f"{sum(stats_boxes_rej_scene_rej.values())} | {stats_boxes_rej_scene_rej}\n")

    end_time_main = time.time()
    elapsed_time = (end_time_main - start_time_main)
    lines.append(f"\nAverage scene generation time (accepted scenes only - {tot_scene_accepted}): {(elapsed_time / tot_scene_accepted):.2f} seconds \n")
    total_scenes = tot_scene_accepted + tot_scene_modified + tot_scene_rejected
    lines.append(f"Average scene generation time (accepted, modified, and rejected scenes - {total_scenes}): "
                 f"{(elapsed_time / total_scenes):.2f} seconds \n")
    lines.append(f"\nTotal execution time since start: {(elapsed_time / 3600):.2f} hours\n\n")

    # If a previous report exists, accumulate global statistics with current job stats
    if report_file_path_previous_exists:
        lines.append("GLOBAL STATS ACCUMULATED:\n\n")
        total_scenes_global += total_scenes
        tot_scene_accepted_global += tot_scene_accepted
        tot_scene_modified_global += tot_scene_modified
        tot_scene_rejected_global += tot_scene_rejected

        # Accumulate instance statistics for generated, modified, and rejected images
        for key, value in stats_boxes_ok_scene_gen.items():
            stats_boxes_ok_scene_gen_global[key] = stats_boxes_ok_scene_gen_global.get(key, 0) + value
        for key, value in stats_boxes_ok_scene_mod.items():
            stats_boxes_ok_scene_mod_global[key] = stats_boxes_ok_scene_mod_global.get(key, 0) + value
        for key, value in stats_boxes_rej_scene_mod.items():
            stats_boxes_rej_scene_mod_global[key] = stats_boxes_rej_scene_mod_global.get(key, 0) + value
        for key, value in stats_boxes_ok_scene_rej.items():
            stats_boxes_ok_scene_rej_global[key] = stats_boxes_ok_scene_rej_global.get(key, 0) + value
        for key, value in stats_boxes_rej_scene_rej.items():
            stats_boxes_rej_scene_rej_global[key] = stats_boxes_rej_scene_rej_global.get(key, 0) + value

        tot_scene_rej_with_rej_global += tot_scene_rej_with_rej
        tot_scene_rej_without_rej_global += tot_scene_rej_without_rej

        lines.append(f"Total scenes attempted: {total_scenes_global} "
                     f"({tot_scene_accepted_global} accepted | {tot_scene_modified_global} modified | {tot_scene_rejected_global} rejected)\n")
        lines.append(f"\nTotal instances accepted in {tot_scene_accepted_global} images generated: {sum(stats_boxes_ok_scene_gen_global.values())} | {stats_boxes_ok_scene_gen_global}\n\n")
        lines.append(f"Total instances accepted in {tot_scene_modified_global} images modified: {sum(stats_boxes_ok_scene_mod_global.values())} | {stats_boxes_ok_scene_mod_global}\n")
        lines.append(f"Total instances rejected in {tot_scene_modified_global} images modified: {sum(stats_boxes_rej_scene_mod_global.values())} | {stats_boxes_rej_scene_mod_global}\n\n")
        lines.append(f"Total instances accepted in {tot_scene_rejected_global} images rejected: {sum(stats_boxes_ok_scene_rej_global.values())} | {stats_boxes_ok_scene_rej_global}\n")
        lines.append(f"Total instances rejected in {tot_scene_rejected_global} images rejected (with [{tot_scene_rej_with_rej_global}] / without [{tot_scene_rej_without_rej_global}] rejected instances): "
                     f"{sum(stats_boxes_rej_scene_rej_global.values())} | {stats_boxes_rej_scene_rej_global}\n")

        # Calculate combined average times and total execution time
        avg_time_accepted_global = ((elapsed_time / tot_scene_accepted) + avg_time_accepted_global) / 2
        avg_time_all_global = ((elapsed_time / total_scenes) + avg_time_all_global) / 2
        total_exec_time_global = (elapsed_time / 3600) + total_exec_time_global

        lines.append(f"\nAverage scene generation time (accepted scenes only - {tot_scene_accepted_global}): {avg_time_accepted_global:.2f} seconds \n")
        lines.append(f"Average scene generation time (accepted, modified, and rejected scenes - {total_scenes_global}): {avg_time_all_global:.2f} seconds \n")
        lines.append(f"\nTotal execution time since start: {total_exec_time_global:.2f} hours\n\n")

    with open(rep_fn, 'a', newline='') as rep_file:
        rep_file.writelines(lines)

    # Ensure Blender is properly closed before exiting
    try:
        import bpy
        bpy.ops.wm.quit_blender()  # Properly shuts down Blender
    except Exception as e:
        print(f"[WARNING] Could not close Blender properly: {e}")

    current_time = datetime.now().strftime("%H:%M:%S")
    print(f"\n[{current_time}][INFO] Exiting script...")
    os._exit(0)  # A safer alternative to sys.exit(0)

# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    global start_time_main, tot_scene_accepted_global, rep_fn
    global total_exec_time_global, avg_time_all_global, avg_time_accepted_global
    global tot_scene_rej_with_rej_global, tot_scene_rej_without_rej_global
    global stats_boxes_rej_scene_rej_global, stats_boxes_ok_scene_rej_global
    global stats_boxes_rej_scene_mod_global, tot_scene_modified_global, stats_boxes_ok_scene_mod_global
    global stats_boxes_ok_scene_gen_global, total_scenes_global, tot_scene_accepted_global, tot_scene_rejected_global

    start_time_main = time.time()

    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-cfp", "--config_file_path", required=True, help="Path to the YAML configuration file.")
    args = parser.parse_args()

    # Load configuration from YAML file
    config = load_config(args.config_file_path)

    # Validate exclusive parameters and camera settings
    validate_exclusive_args(config, "min_instances", "total_images")
    validate_min_instances(config["min_instances_accepted"], config["num_instances"])
    camera_positions = validate_camera_positions(config["camera_position"])
    validate_rotation_constraints(camera_positions, config["camera_rotation_constraints"])
    validate_focal_length_constraints(camera_positions, config["focal_length_constraints"])
    validate_lightning_constraints(config["lightning_constraints"])

    if not config["prism_types"]:
        print("List of 'prism_types' cannot be empty. Exiting.")
        sys.exit(0)

    # Load prism dictionary from file
    dict_prisms = {}
    dict_file_path = os.path.join(config["dictionary_cubes"], "dict_boxes.txt")
    with open(dict_file_path, "r") as f:
        for line in f:
            key, value = line.strip().split(":")
            dict_prisms[key.strip()] = [float(x) for x in value.strip().strip("[]").split(",")]

    # Verify each prism type exists in the dictionary
    for prism in config["prism_types"]:
        if prism and prism not in dict_prisms:
            print(f"Prism type {prism} not found in dictionary {config['dictionary_cubes']}")
            sys.exit(0)

    # Initialize color dictionaries and type indices
    config["prism_types"].sort()  # Sort classes alphabetically
    boxes_types = []
    assets_obj = [file for file in os.listdir(config["assets"]) if file.endswith('.obj')]
    dict_assets = {}
    for asset in assets_obj:
        filepath = os.path.join(config["assets"], asset)
        dimensions = get_obj_dimensions(filepath)
        x, y, z = dimensions
        dict_assets[asset[:-4]] = [float(t) for t in [x, z, y]]
    color_dict_lab, lab_dict_color = {}, {}
    for i, label in enumerate(config["prism_types"] + assets_obj):
        label_name = label[:-4] if label.endswith('.obj') else label
        boxes_types.append(label_name)
        gr_lv = (i + 1) / len(config["prism_types"] + assets_obj)
        color_dict_lab[label] = (gr_lv, gr_lv, gr_lv)
        lab_dict_color[int(255 * gr_lv)] = label

    # Setup output directories and initialize report/CSV/JSON files
    base_output_dir, reset_flag = setup_output_directory(config["output_dir"])
    report_file_path = os.path.join(base_output_dir, "report_info.txt")
    if os.path.exists(report_file_path):
        global report_file_path_previous_exists
        report_file_path_previous_exists = True
        load_previous_stats(report_file_path)

    data_dir = os.path.join(base_output_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    rep_fn = os.path.join(base_output_dir, "report_info.txt")
    if reset_flag or not os.path.isfile(rep_fn):
        with open(rep_fn, 'w', newline='') as rep_file:
            rep_file.write("-" * 115 + "\n")

    csv_glbl_fn = os.path.join(base_output_dir, "data_info.csv")
    if reset_flag or not os.path.isfile(csv_glbl_fn):
        with open(csv_glbl_fn, 'w', newline='') as csv_glbl_file:
            csv_glbl_file.write("")
    write_glbl_csv_header = reset_flag or not os.path.isfile(csv_glbl_fn)

    json_path = os.path.join(base_output_dir, "pose_estimation.json")
    if reset_flag and os.path.exists(json_path):
        os.remove(json_path)
    if not os.path.exists(json_path):
        data_json = {"images": {}}
    else:
        with open(json_path, 'r') as f:
            data_json = json.load(f)

    # Get .obj or .ply files from the specified directory
    directory = config["directory"]
    obj_files = [file for file in os.listdir(directory) if file.endswith((".obj", ".ply"))]
    if not obj_files:
        print("No .obj or .ply files found in the specified directory. Exiting.")
        sys.exit(0)

    # Generate unique colors for each instance
    colors_dict, used_colors = {}, set()
    for color_id in range(config["num_instances"]):
        colors_dict[color_id] = generate_unique_color(used_colors)

    type_index_dict = {name: index + 1 for index, name in enumerate(boxes_types)}

    # Import a random .obj/.ply file
    obj_file = random.choice(obj_files)
    #bpy.ops.wm.ply_import(filepath=os.path.join(directory, obj_file))
    """ Imports file silently, suppressing terminal output. """
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        bpy.ops.wm.ply_import(filepath=os.path.join(directory, obj_file))
    imported_objects = bpy.context.selected_objects
    ply_obj = imported_objects[0]
    ply_obj_name = ply_obj.name

    # Set render dimensions
    set_render_dimensions(config["img_width"], config["img_height"])

    # Initialize bounding box statistics for generated, modified, and rejected scenes
    (stats_boxes_ok_scene_rej, stats_boxes_rej_scene_rej,
     stats_boxes_ok_scene_mod, stats_boxes_rej_scene_mod,
     stats_boxes_ok_scene_gen, stats_boxes_rej_scene_gen) = initialize_stats(config["prism_types"], assets_obj)

    # Create a BVH tree from the imported object to avoid recomputation each iteration
    ply_obj = bpy.data.objects[ply_obj_name]
    bm_ply_obj = bmesh.new()
    bm_ply_obj.from_mesh(ply_obj.data)
    bm_ply_obj.transform(ply_obj.matrix_world)
    obj_ply_BVHtree = BVHTree.FromBMesh(bm_ply_obj)

    global tot_scene_rejected; tot_scene_rejected = 0
    global tot_scene_modified; tot_scene_modified = 0
    global tot_scene_rej_with_rej; tot_scene_rej_with_rej = 0
    global tot_scene_rej_without_rej; tot_scene_rej_without_rej = 0
    global tot_scene_accepted; tot_scene_accepted = 0

    try:
        total_images = config.get("total_images", None)
        min_instances_total = config.get("min_instances", None)
        if total_images is not None:
            # Loop to render a fixed number of images
            for i in range(tot_scene_accepted_global, config["total_images"]):
                (tot_scene_rejected, tot_scene_modified,
                tot_scene_rej_with_rej, tot_scene_rej_without_rej,
                write_glbl_csv_header) = generate_scene(
                    assets_obj, obj_files, dict_prisms, dict_assets, data_dir, data_json,
                    type_index_dict, boxes_types, stats_boxes_ok_scene_rej, stats_boxes_rej_scene_rej,
                    stats_boxes_ok_scene_mod, stats_boxes_rej_scene_mod,
                    stats_boxes_ok_scene_gen, stats_boxes_rej_scene_gen, csv_glbl_fn,
                    tot_scene_rejected, tot_scene_modified, tot_scene_rej_with_rej, tot_scene_rej_without_rej,
                    write_glbl_csv_header, ply_obj_name, config, i, obj_ply_BVHtree, json_path
                )
                tot_scene_accepted = i + 1 - tot_scene_accepted_global
                if tot_scene_accepted % 50 == 0:
                    current_time = datetime.now().strftime("%H:%M:%S")
                    print(f"\n[{current_time}][INFO] Scene {tot_scene_accepted_global + tot_scene_accepted} accepted "
                          f"({tot_scene_accepted} in current execution).\n")
        elif min_instances_total is not None:
            # Loop until the minimum instance condition is met
            min_instances = False
            if tot_scene_accepted_global == 0:
                img_id = -1
            else:
                img_id = tot_scene_accepted_global - 1

            while not min_instances:
                img_id += 1
                (tot_scene_rejected, tot_scene_modified,
                tot_scene_rej_with_rej, tot_scene_rej_without_rej,
                write_glbl_csv_header) = generate_scene(
                    assets_obj, obj_files, dict_prisms, dict_assets, data_dir, data_json,
                    type_index_dict, boxes_types, stats_boxes_ok_scene_rej, stats_boxes_rej_scene_rej,
                    stats_boxes_ok_scene_mod, stats_boxes_rej_scene_mod,
                    stats_boxes_ok_scene_gen, stats_boxes_rej_scene_gen, csv_glbl_fn,
                    tot_scene_rejected, tot_scene_modified, tot_scene_rej_with_rej, tot_scene_rej_without_rej,
                    write_glbl_csv_header, ply_obj_name, config, img_id, obj_ply_BVHtree, json_path
                )
                # Check if the combined statistics meet the minimum instances requirement
                combined_stats = {}
                if stats_boxes_ok_scene_gen_global is not None:
                    for key in stats_boxes_ok_scene_gen:
                        combined_stats[key] = stats_boxes_ok_scene_gen.get(key, 0) + stats_boxes_ok_scene_gen_global.get(key, 0)
                    if min(combined_stats.values()) >= config["min_instances"]:
                        min_instances = True
                else:
                    if min(stats_boxes_ok_scene_gen.values()) >= config["min_instances"]:
                        min_instances = True

                tot_scene_accepted += 1
                if tot_scene_accepted % 50 == 0:
                    current_time = datetime.now().strftime("%H:%M:%S")
                    stats_to_log = stats_boxes_ok_scene_gen if stats_boxes_ok_scene_gen_global is None else combined_stats
                    print(f"\n[{current_time}][INFO] Scene {tot_scene_accepted_global + tot_scene_accepted} accepted "
                          f"({tot_scene_accepted} in current execution).\nTotal instances per class: {stats_to_log}.\n")
    except KeyboardInterrupt:
        print("KeyboardInterrupt detected. Exiting...")
    finally:
        # Ensure stats are printed only once before exiting
        write_stats_signal_handler(None, None)

    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, write_stats_signal_handler)  # Ctrl+C
    signal.signal(signal.SIGTERM, write_stats_signal_handler)  # kill PID
    main()
