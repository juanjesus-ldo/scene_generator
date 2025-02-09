#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Monitoring main generator script.
Usage: time ./monitoring_generator.py -restart_interval 10 -folder_images /home/juanje/Escritorio/data_generator_refactorized/SYNTHETIC_DATA_GENERATION
"""

import argparse
import os
import re
import subprocess
import time
import signal
import sys

from datetime import datetime

def get_latest_image_id(folder_path):
    """
    Searches for the highest image ID in the specified folder by matching filenames.

    Args:
        folder_path (str): Path to the folder containing image subfolder "data".

    Returns:
        int: The highest image ID found, or -1 if no images are found.
    """
    max_id = -1
    pattern = re.compile(r"(\d+)_boxes3d\.png")
    data_folder = os.path.join(folder_path, "data")
    for filename in os.listdir(data_folder):
        match = pattern.match(filename)
        if match:
            image_id = int(match.group(1))
            max_id = max(max_id, image_id)
    return max_id

def handle_termination(signal_number, frame, process):
    """
    Handles termination signals to cleanly terminate both the monitoring and scene generation processes.

    Args:
        signal_number: The signal number.
        frame: The current stack frame.
        process: The subprocess running the scene generation script.
    """
    print("\n[INFO] Terminating both the monitoring and scene generation processes...")
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)  # Esperar hasta 5 segundos para asegurar el cierre
        except subprocess.TimeoutExpired:
            process.kill()  # Forzar cierre si aún está activo
    sys.exit(0)

def main():
    """
    Main loop that periodically checks the monitored folder for new images.
    Restarts the scene generation process if the number of new images meets the restart interval.
    """
    parser = argparse.ArgumentParser(
        description="Script to periodically restart scene generation based on image count."
    )
    parser.add_argument("-restart_interval", type=int, required=True,
                        help="Number of images to generate before restarting the process.")
    parser.add_argument("-folder_images", type=str, required=True,
                        help="Folder path to monitor for images.")
    args = parser.parse_args()

    # Path to the scene generation script and configuration file
    script_path = "./main_generator.py"
    config_file_path = "./config.yaml"

    # Start the scene generation process for the first time
    process = subprocess.Popen(["python3", script_path, "--config_file_path", config_file_path])

    # Set up signal handler for Ctrl+C termination
    signal.signal(signal.SIGINT, lambda sig, frame: handle_termination(sig, frame, process))

    if os.path.exists(args.folder_images):
        last_id_stop = get_latest_image_id(args.folder_images)
    else:
        last_id_stop = 0  # If no images exist yet, start from 0

    while True:
        # Wait 30 seconds before checking the folder again
        time.sleep(30)

        # If the process has ended naturally, exit the monitoring script
        if process.poll() is not None:
            current_time = datetime.now().strftime("%H:%M:%S")
            # Ensure that `write_stats_signal_handler()` runs before terminating
            # Send SIGINT to the process to trigger `write_stats_signal_handler()` within `main_generator.py`
            process.send_signal(signal.SIGINT)
            process.wait()  # Ensure the process finishes before exiting
            print(f"\n[{current_time}][INFO] The scene generation script has finished. Exiting monitoring script.")
            break

        # Get the latest image ID from the monitored folder
        latest_image_id = get_latest_image_id(args.folder_images)
        current_time = datetime.now().strftime("%H:%M:%S")
        if latest_image_id == -1:
            continue

        # If the number of new images reaches the restart interval, restart the process
        if latest_image_id - last_id_stop >= args.restart_interval:
            current_time = datetime.now().strftime("%H:%M:%S")
            print(f"\n[{current_time}][INFO] Restarting the process...")
            last_id_stop = latest_image_id
            process.send_signal(signal.SIGINT)
            process.wait() # Wait for the process to fully terminate
            # Restart the process
            process = subprocess.Popen(["python3", script_path, "--config_file_path", config_file_path])
            current_time = datetime.now().strftime("%H:%M:%S")
            print(f"\n[{current_time}][INFO] Process restarted from image with ID {latest_image_id}.")

if __name__ == "__main__":
    main()
