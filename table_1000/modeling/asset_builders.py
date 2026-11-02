"""Small Blender modeling helpers shared by the stationery asset sources.

Generated source snapshots import these helpers from the repository package.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import shutil
import sys

import bpy
import bmesh


def material(name, color, roughness=0.35, metallic=0.0, transmission=0.0):
    surface = bpy.data.materials.new(name)
    surface.use_nodes = True
    shader = surface.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = (*color, 1)
    shader.inputs["Roughness"].default_value = roughness
    shader.inputs["Metallic"].default_value = metallic
    shader.inputs["Transmission Weight"].default_value = transmission
    shader.inputs["IOR"].default_value = 1.46
    surface.diffuse_color = (*color, 1)
    return surface


def asset(name, origin, axes, front):
    root = bpy.data.objects.new(name, None)
    bpy.context.scene.collection.objects.link(root)
    root["coordinate_frame_normalized"] = True
    root["origin_convention"] = origin
    root["axis_convention"] = axes
    root["semantic_front"] = front
    root.empty_display_size = 0.02
    return root


def rigid_settings(obj, shape):
    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.rigidbody.object_add()
    obj.select_set(False)
    obj.rigid_body.collision_shape = shape
    obj.rigid_body.mesh_source = "BASE"
    obj.rigid_body.use_margin = True
    obj.rigid_body.collision_margin = 0.0


def body(name, root):
    obj = bpy.data.objects.new(name, bpy.data.meshes.new(name + "_frame"))
    bpy.context.scene.collection.objects.link(obj)
    obj.parent = root
    obj["rigid_body_root"] = True
    rigid_settings(obj, "COMPOUND")
    return obj


def deformable(name, root):
    """Stage-one deforming geometry owner, not a physical rigid body."""
    obj = bpy.data.objects.new(name, bpy.data.meshes.new(name + "_frame"))
    bpy.context.scene.collection.objects.link(obj)
    obj.parent = root
    obj["deformable_root"] = True
    return obj


def part(name, root):
    """Semantic component; Compound only when owned by a rigid body."""
    if root.get("deformable_root"):
        obj = bpy.data.objects.new(name, bpy.data.meshes.new(name + "_frame"))
        bpy.context.scene.collection.objects.link(obj)
        obj.parent = root
    else:
        obj = body(name, root)
        del obj["rigid_body_root"]
    obj["semantic_part"] = True
    return obj


def group_part(name, root, visuals, colliders):
    """Put authored geometry under a part without changing its saved world pose."""
    bpy.context.view_layer.update()
    node = part(name, root)
    bpy.context.view_layer.update()
    for role, objects in [("visual", visuals), ("collision", colliders)]:
        for i, obj in enumerate(objects, 1):
            world = obj.matrix_world.copy()
            obj.parent = node
            obj.matrix_world = world
            obj.name = name + "." + role + (str(i) if len(objects) > 1 else "")
    return node


def mesh(name, vertices, faces, parent, surface=None, shape=None):
    data = bpy.data.meshes.new(name + "_mesh")
    data.from_pydata(vertices, [], faces)
    data.update()
    bm = bmesh.new()
    bm.from_mesh(data)
    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    bm.to_mesh(data)
    bm.free()
    obj = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(obj)
    obj.parent = parent
    if shape:
        obj["geometry_role"] = "collision"
        obj.hide_render = True
        obj.display_type = "WIRE"
        rigid_settings(obj, shape)
    else:
        obj["geometry_role"] = "visual"
        data.materials.append(surface)
    return obj


def collider(visual, shape="CONVEX_HULL"):
    proxy = bpy.data.objects.new("Collision_" + visual.name, visual.data.copy())
    bpy.context.scene.collection.objects.link(proxy)
    proxy.parent = visual.parent
    proxy.matrix_basis = visual.matrix_basis.copy()
    proxy["geometry_role"] = "collision"
    proxy.hide_render = True
    proxy.display_type = "WIRE"
    rigid_settings(proxy, shape)
    return proxy


BOX_FACES = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1),
             (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]


def box(name, dimensions, position, surface, parent, bevel=0.0, collision=True):
    vertices = [(x * dimensions[0] / 2, y * dimensions[1] / 2, z * dimensions[2] / 2)
                for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
    obj = mesh(name, vertices, BOX_FACES, parent, surface)
    obj.location = position
    if bevel:
        modifier = obj.modifiers.new("Manufactured edge", "BEVEL")
        modifier.width = bevel
        modifier.segments = 1
        modifier = obj.modifiers.new("Corner normals", "WEIGHTED_NORMAL")
        modifier.keep_sharp = True
    if collision:
        collider(obj, "BOX")
    return obj


def axis_point(axial, u, v, axis):
    if axis == "X":
        return (axial, u, v)
    if axis == "Y":
        return (u, axial, v)
    return (u, v, axial)


def cylinder(name, radius, depth, position, surface, parent, radius_end=None,
             axis="X", segments=16, collision=True):
    radius_end = radius if radius_end is None else radius_end
    vertices = [axis_point(end, r * math.cos(i * math.tau / segments),
                           r * math.sin(i * math.tau / segments), axis)
                for end, r in ((-depth / 2, radius), (depth / 2, radius_end))
                for i in range(segments)]
    faces = [tuple(range(segments - 1, -1, -1)), tuple(range(segments, 2 * segments))]
    faces.extend((i, (i + 1) % segments, (i + 1) % segments + segments, i + segments)
                 for i in range(segments))
    obj = mesh(name, vertices, faces, parent, surface)
    obj.location = position
    for face in obj.data.polygons:
        face.use_smooth = len(face.vertices) == 4
    if collision:
        collider(obj)
    return obj


def tube(name, outer, inner, depth, position, surface, parent, axis="X",
         outer_end=None, inner_end=None, segments=16, collision=True, solid_collision=False):
    """Hollow visual tube with closed convex sectors, or one solid outer collision hull.

    Use ``solid_collision`` only when the cavity has no function (a fixed pen
    barrel occupied by its refill); containers and removable caps stay hollow.
    Future moving button/refill mechanisms need separate jointed rigid bodies,
    with collision disabled between the connected bodies.
    """
    outer_end = outer if outer_end is None else outer_end
    inner_end = inner if inner_end is None else inner_end
    vertices = [axis_point(end, r * math.cos(i * math.tau / segments),
                           r * math.sin(i * math.tau / segments), axis)
                for end, r in ((-depth / 2, outer), (depth / 2, outer_end),
                               (-depth / 2, inner), (depth / 2, inner_end))
                for i in range(segments)]
    faces = []
    for i in range(segments):
        j = (i + 1) % segments
        sector_faces = [(i, j, segments + j, segments + i),
                        (2 * segments + j, 2 * segments + i, 3 * segments + i, 3 * segments + j),
                        (i, 2 * segments + i, 2 * segments + j, j),
                        (segments + i, segments + j, 3 * segments + j, 3 * segments + i)]
        faces.extend(sector_faces)
        if collision and not solid_collision:
            indices = [i, j, segments + i, segments + j,
                       2 * segments + i, 2 * segments + j, 3 * segments + i, 3 * segments + j]
            local = {index: n for n, index in enumerate(indices)}
            closed = sector_faces + [(i, segments + i, 3 * segments + i, 2 * segments + i),
                                     (j, 2 * segments + j, 3 * segments + j, segments + j)]
            proxy = mesh(f"Collision_{name}_{i:02d}", [vertices[k] for k in indices],
                         [tuple(local[k] for k in face) for face in closed], parent,
                         shape="CONVEX_HULL")
            proxy.location = position
    if collision and solid_collision:
        closed = [tuple(range(segments - 1, -1, -1)), tuple(range(segments, 2 * segments))]
        closed.extend((i, (i + 1) % segments, segments + (i + 1) % segments, segments + i)
                      for i in range(segments))
        proxy = mesh(f"Collision_{name}", vertices[:2 * segments], closed, parent,
                     shape="CONVEX_HULL")
        proxy.location = position
    obj = mesh(name, vertices, faces, parent, surface)
    obj.location = position
    for i, face in enumerate(obj.data.polygons):
        face.use_smooth = i % 4 in (0, 1)
    return obj


def label_patch(name, radius, depth, position, surface, parent, angle=math.pi / 2,
                span=1.12, segments=4, axis="X"):
    """A thin closed label following a cylindrical surface."""
    count = segments + 1
    vertices = [axis_point(x, r * math.cos(angle - span / 2 + i * span / segments),
                          r * math.sin(angle - span / 2 + i * span / segments), axis)
                for x, r in ((-depth / 2, radius), (depth / 2, radius),
                             (-depth / 2, radius - 0.00005), (depth / 2, radius - 0.00005))
                for i in range(count)]
    faces = []
    for i in range(segments):
        j = i + 1
        faces.extend([(i, j, count + j, count + i),
                      (2 * count + j, 2 * count + i, 3 * count + i, 3 * count + j),
                      (i, 2 * count + i, 2 * count + j, j),
                      (count + i, count + j, 3 * count + j, 3 * count + i)])
    for i in (0, segments):
        faces.append((i, count + i, 3 * count + i, 2 * count + i))
    obj = mesh(name, vertices, faces, parent, surface)
    obj.location = position
    for i, face in enumerate(obj.data.polygons):
        face.use_smooth = i % 4 in (0, 1)
    return obj


def capped_pen(profile):
    """Two top-level rigid bodies; authored visual/collision clearance agrees."""
    g = profile["geometry"]
    n, seat = g["segments"], g["seat_x_m"]
    barrel_mat = material("Barrel plastic", profile["barrel_color"], 0.35)
    cap_mat = material("Cap plastic", profile["cap_color"], 0.30,
                       transmission=profile["cap_transmission"])
    ink = material("Writing tip", (0.018, 0.024, 0.020), 0.35)
    pen, cap = body("body", None), body("cap", None)
    pen["coordinate_frame_normalized"] = True
    pen["origin_convention"] = "barrel reference center; cap local origin at mouth"
    pen["axis_convention"] = "+X rear; -X writing tip and cap removal; Y/Z radial"
    pen["semantic_front"] = "writing tip at -X"
    for root, mass in [(pen, profile["body_mass_kg"]), (cap, profile["cap_mass_kg"])]:
        root.rigid_body.mass = mass
    cap.location.x = seat
    tip = seat - g["neck_length_m"]
    cylinder("Barrel", g["barrel_radius_m"], g["barrel_end_x_m"]-seat-0.00005,
             ((seat+0.00005+g["barrel_end_x_m"])/2,0,0), barrel_mat, pen, segments=n)
    if "rear_end_x_m" in g:
        rear_mat = material("Rear plug", profile["rear_color"], 0.34,
                            transmission=profile["rear_transmission"])
        cylinder("Rear plug", g["rear_radius_m"], g["rear_end_x_m"]-g["rear_start_x_m"],
                 ((g["rear_start_x_m"]+g["rear_end_x_m"])/2,0,0), rear_mat, pen, segments=n)
    if "label_length_m" in g:
        label = material("Paper label", profile["label_color"], 0.60)
        tube("Pale paper label", g["barrel_radius_m"]+.00005, g["barrel_radius_m"],
             g["label_length_m"], (seat+.0001+g["label_length_m"]/2,0,0),
             label, pen, segments=n, collision=False)
    cylinder("Neck", g["neck_radius_m"], g["neck_length_m"]+0.00005,
             ((tip+seat+0.00005)/2,0,0), barrel_mat, pen, segments=n)
    cylinder("Nib", g["nib_tip_radius_m"], g["nib_length_m"],
             (tip-g["nib_length_m"]/2,0,0), ink, pen,
             radius_end=g["nib_base_radius_m"], segments=n)
    if profile["metal_point"]:
        steel = material("Fine steel point", (0.50,0.54,0.57), 0.24, metallic=0.90)
        cylinder("Fine point", 0.0003, 0.0035, (tip-g["nib_length_m"]-0.00175,0,0),
                 steel, pen, radius_end=0.0007, segments=12)
    inner_end = -g["cap_length_m"]+g["end_thickness_m"]
    lip_end = -g["lip_length_m"]
    # Circumscribed inner polygons guarantee the requested minimum radial
    # clearance even when the neck and cap polygons rotate relative to one another.
    inner = (g["neck_radius_m"]+g["rigid_lip_clearance_m"])/math.cos(math.pi/n)
    tube("Cap shell", g["cap_radius_m"], g["shell_inner_radius_m"]/math.cos(math.pi/n),
         lip_end-inner_end, ((inner_end+lip_end)/2,0,0), cap_mat, cap, segments=n)
    tube("Cap lip", g["cap_radius_m"], inner, -lip_end,
         (lip_end/2,0,0), cap_mat, cap, segments=n)
    cylinder("Cap end", g["cap_radius_m"], g["end_thickness_m"],
             (-g["cap_length_m"]+g["end_thickness_m"]/2,0,0), cap_mat, cap, segments=n)
    if "clip_dimensions_m" in profile:
        clip_mat = material("Pocket clip", profile["clip_color"], 0.32)
        box("Cap clip", profile["clip_dimensions_m"], profile["clip_position_m"],
            clip_mat, cap, bevel=0.0003)


def joint(name, kind, fixed, moving, position, limits, rotation=(0, 0, 0)):
    obj = bpy.data.objects.new(name, None)
    bpy.context.scene.collection.objects.link(obj)
    obj.location = position
    obj.rotation_euler = rotation
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.rigidbody.constraint_add()
    obj.select_set(False)
    c = obj.rigid_body_constraint
    c.type = kind
    c.object1, c.object2 = fixed, moving
    c.disable_collisions = False
    if kind == "SLIDER":
        c.use_limit_lin_x = True
        c.limit_lin_x_lower, c.limit_lin_x_upper = limits
    else:
        c.use_limit_ang_z = True
        c.limit_ang_z_lower, c.limit_ang_z_upper = limits
    obj["preview_joint"] = True
    return obj


def generate(build, source_file, argv=None):
    parser = argparse.ArgumentParser(description="Build this object and a reusable source snapshot")
    parser.add_argument("--output", type=Path, required=True)
    if argv is None:
        argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    output = parser.parse_args(argv).output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0
    scene.unit_settings.length_unit = "METERS"
    scene.frame_start, scene.frame_end = 0, 48
    build()
    if scene.rigidbody_world:
        scene.rigidbody_world.enabled = False
    scene["penetration_tolerance_m"] = 0.0002
    scene.frame_set(0)
    bpy.context.view_layer.update()
    bpy.context.preferences.filepaths.save_version = 0
    destination = output / "object.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(destination))
    source = Path(source_file).resolve()
    originals = [source, *(source.parent / name for name in ("metadata.json", "preview.json", "README.md"))]
    thumbnail = source.parent / "three_quarter.jpg"
    if thumbnail.exists():
        originals.append(thumbnail)
    for original in originals:
        target = output / original.name
        if original != target:
            shutil.copyfile(original, target)
    print("MODEL_BUILT", destination, flush=True)
