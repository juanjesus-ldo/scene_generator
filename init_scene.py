import bpy
import os

from camera import init_camera
from utils import stdout_redirected, stderr_redirected

def init_scene(ply_obj_name):
    """
    Initialize the scene by deleting all objects except the specified PLY object,
    and set up the Cycles render engine with GPU support.

    Args:
        ply_obj_name (str): The name of the PLY object to preserve.
    """
    # Deselect all objects
    bpy.ops.object.select_all(action='DESELECT')

    # Select all objects except the PLY object and delete them
    for obj in bpy.data.objects:
        if obj.name != ply_obj_name:
            obj.select_set(True)
    bpy.ops.object.delete()

    # Set render engine to Cycles
    bpy.data.scenes[0].render.engine = "CYCLES"

    with stdout_redirected(), stderr_redirected():
        # Set the device type to OPTIX (or OPENCL)
        bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "OPTIX"

        # Configure Cycles to use GPU and the supported feature set
        bpy.context.scene.cycles.device = "GPU"
        bpy.context.scene.cycles.feature_set = "SUPPORTED"

        # Detect GPU devices
        bpy.context.preferences.addons["cycles"].preferences.get_devices()
        print(bpy.context.preferences.addons["cycles"].preferences.compute_device_type)
        for d in bpy.context.preferences.addons["cycles"].preferences.devices:
            if 'Intel' in d['name']:
                continue
            if 'CPU' in d['name']:
                continue
            d["use"] = 1  # Use this device
            print(d["name"], d["use"])

    bpy.context.scene.cycles.adaptive_threshold = 0.05
    bpy.context.scene.cycles.samples = 2048


def load_scene_and_init_camera(obj_file, directory, obj_f, blend_f, ply_f, ply_obj_name,
                               camera_position, camera_rotation_constraints, focal_length_constraints):
    """
    Load a scene file (.obj, .blend, or .ply) and initialize the camera.

    Depending on the file type, the function loads the corresponding file,
    sets up the scene collection (if a .blend file) or applies a vertex-color material (for .ply files),
    and then calls init_camera to set up the camera.

    Args:
        obj_file (str): Filename of the scene file.
        directory (str): Directory where the file is located.
        obj_f (bool): True if the file is an .obj file.
        blend_f (bool): True if the file is a .blend file.
        ply_f (bool): True if the file is a .ply file.
        ply_obj_name (str): The name of the PLY object to use.
        camera_position (list): List of possible camera positions.
        camera_rotation_constraints (list): List of rotation constraints for the camera.
        focal_length_constraints (list): List of focal length constraints for the camera.

    Returns:
        Tuple: (loaded object or collection, random index, camera position string, camera rotation string, focal length string)
    """
    if obj_f:
        # Load .obj file
        with stdout_redirected(), stderr_redirected():
            bpy.ops.wm.obj_import(filepath=os.path.join(directory, obj_file),
                                  directory=directory,
                                  files=[{"name": obj_file, "name": obj_file}])
        obj = bpy.context.object  # Assume the imported object is active
        _, random_index, camera_pos, camera_rot, focal_length = init_camera(
            camera_position, camera_rotation_constraints, focal_length_constraints, obj)
        return obj, random_index, camera_pos, camera_rot, focal_length

    elif blend_f:
        # Load .blend file
        bpy.ops.wm.open_mainfile(filepath=os.path.join(directory, obj_file))

        # Create a new collection and add all scene objects to it
        scene_collection_name = "MySceneCollection"
        scene_collection = bpy.data.collections.new(scene_collection_name)
        for obj in bpy.context.scene.objects:
            scene_collection.objects.link(obj)

        # Link the new collection to the main collection
        main_collection = bpy.context.scene.collection
        main_collection.children.link(scene_collection)

        # Select all objects in the new collection
        bpy.context.view_layer.objects.active = None
        for obj in scene_collection.objects:
            obj.select_set(True)

        # Get the center of the collection, the location of the object closest to the center,
        # and the midpoint of the maximum height (assumed to be provided by external functions)
        center = get_center(scene_collection)
        closest_object_location = get_closest_object_to_center(scene_collection, center)
        midpoint_z = get_collection_midpoint(scene_collection)

        _, random_index, camera_pos, camera_rot, focal_length = init_camera(
            camera_position, camera_rotation_constraints, focal_length_constraints,
            scene_collection, closest_object_location, midpoint_z)
        return scene_collection, random_index, camera_pos, camera_rot, focal_length

    elif ply_f:
        # Load .ply file by getting the object with the specified name
        ply_obj = bpy.data.objects.get(ply_obj_name)
        bpy.context.view_layer.objects.active = ply_obj
        ply_obj.select_set(True)
        obj = bpy.context.object  # Assume the imported object is active

        # Disable unnecessary visibility properties for better performance
        bpy.context.object.visible_glossy = False
        bpy.context.object.visible_transmission = False
        bpy.context.object.visible_volume_scatter = False
        bpy.context.object.visible_shadow = False

        # Create a material to display vertex colors (since Blender does not interpret them by default)
        material = bpy.data.materials.new(name="VertexColorMaterial")
        material.use_nodes = True
        nodes = material.node_tree.nodes

        # Remove the default Principled BSDF node if it exists
        principled_bsdf = nodes.get("Principled BSDF")
        if principled_bsdf:
            nodes.remove(principled_bsdf)

        # Add an Attribute node and an Emission node, then link them
        attribute_node = nodes.new(type='ShaderNodeAttribute')
        attribute_node.attribute_name = "Col"
        emission_node = nodes.new(type='ShaderNodeEmission')
        material.node_tree.links.new(attribute_node.outputs['Color'], emission_node.inputs['Color'])

        # Link the Emission node to the Material Output node
        output_node = nodes.get("Material Output")
        material.node_tree.links.new(emission_node.outputs['Emission'], output_node.inputs['Surface'])

        # Assign the material to the object
        if len(obj.data.materials) == 0:
            obj.data.materials.append(material)
        else:
            obj.data.materials[0] = material

        # Set the 3D viewport shading to 'MATERIAL' for proper display
        for area in bpy.context.screen.areas:
            if area.type == 'VIEW_3D':
                for space in area.spaces:
                    if space.type == 'VIEW_3D':
                        space.shading.type = 'MATERIAL'
                        break
                break

        _, random_index, camera_pos, camera_rot, focal_length = init_camera(
            camera_position, camera_rotation_constraints, focal_length_constraints, obj)
        return obj, random_index, camera_pos, camera_rot, focal_length
