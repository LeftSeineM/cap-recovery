"""Accepted pen profiles built with the shared package and semantic parts."""
import math

import bpy

from table_1000.modeling.asset_builders import (
    BOX_FACES, asset, body, box, capped_pen, collider, cylinder, group_part,
    joint, label_patch, material, mesh, tube,
)

SEGMENTS = 12


def organize_parts(specification):
    """Group functional geometry, preserving world poses and hollow sectors."""
    for root_name, part_name, visual_names in specification:
        root = bpy.data.objects[root_name]
        visuals = [bpy.data.objects[name] for name in visual_names]
        collisions = sorted(
            [obj for obj in root.children if any(
                obj.name == 'Collision_' + name or obj.name.startswith('Collision_' + name + '_')
                for name in visual_names)], key=lambda obj: obj.name)
        group_part(root_name + '.' + part_name, root, visuals, collisions)


def organize_capped_parts():
    names = [
        ('body', 'barrel', 'Barrel'), ('body', 'rear_plug', 'Rear plug'),
        ('body', 'label', 'Pale paper label'), ('body', 'neck', 'Neck'),
        ('body', 'tip', 'Nib'), ('body', 'tip', 'Chisel felt tip'),
        ('body', 'steel_point', 'Fine point'), ('body', 'label', 'Flat face label'),
        ('body', 'shoulder_band', 'Black shoulder band'),
        ('cap', 'shell', 'Cap shell'), ('cap', 'lip', 'Cap lip'),
        ('cap', 'end', 'Cap end'), ('cap', 'clip', 'Cap clip'),
    ]
    specification = [(root, part, [visual]) for root, part, visual in names
                     if visual in bpy.data.objects]
    marks = [f'Printed barrel mark{i}' for i in range(1, 5)
             if f'Printed barrel mark{i}' in bpy.data.objects]
    if marks:
        specification.append(('body', 'print', marks))
    organize_parts(specification)


def marker_profile(color):
    """Keep the accepted coarse marker's geometry, with a new barrel color."""
    return {
        "geometry": {
            "segments": SEGMENTS, "seat_x_m": -0.032, "barrel_end_x_m": 0.086,
            "barrel_radius_m": 0.009, "neck_length_m": 0.026,
            "neck_radius_m": 0.0078, "nib_length_m": 0.013,
            "nib_tip_radius_m": 0.0022, "nib_base_radius_m": 0.0042,
            "cap_radius_m": 0.01, "cap_length_m": 0.061,
            "end_thickness_m": 0.0015, "lip_length_m": 0.003,
            "shell_inner_radius_m": 0.0086, "rigid_lip_clearance_m": 0.0002,
            "label_length_m": 0.046,
        },
        "barrel_color": color, "cap_color": [0.67, 0.70, 0.69],
        "cap_transmission": 0.48, "body_mass_kg": 0.012,
        "cap_mass_kg": 0.003, "metal_point": False,
        "label_color": [0.82, 0.82, 0.73],
    }


def highlighter(color):
    """Short rounded rectangular barrel and matching hollow removable cap."""
    profile = {
        "geometry": {
            "segments": SEGMENTS, "seat_x_m": -0.026, "barrel_end_x_m": 0.055,
            "barrel_radius_m": 0.0105, "neck_length_m": 0.013,
            "neck_radius_m": 0.0088, "nib_length_m": 0.006,
            "nib_tip_radius_m": 0.0015, "nib_base_radius_m": 0.003,
            "cap_radius_m": 0.011, "cap_length_m": 0.029,
            "end_thickness_m": 0.0012, "lip_length_m": 0.002,
            "shell_inner_radius_m": 0.0098, "rigid_lip_clearance_m": 0.0003,
        },
        "barrel_color": color, "cap_color": color, "cap_transmission": 0,
        "body_mass_kg": 0.011, "cap_mass_kg": 0.003, "metal_point": False,
    }
    capped_pen(profile)
    # Apply a common homothetic section to both visuals and every convex sector.
    # Scale mesh vertices rather than objects, preserving applied object scales.
    for obj in bpy.context.scene.objects:
        if obj.type != 'MESH':
            continue
        for vertex in obj.data.vertices:
            y, z = vertex.co.y, vertex.co.z
            radial = math.hypot(y, z)
            if radial:
                section = ((y / radial) ** 4 + (z / radial) ** 4) ** -0.25
                vertex.co.y = y * section
                vertex.co.z = z * section * (0.015 / 0.022)
        obj.data.update()
    # A chisel tip has a sloped end face, rather than a round ballpoint.
    for name in ('Nib', 'Collision_Nib'):
        bpy.data.objects.remove(bpy.data.objects[name], do_unlink=True)
    tip_surface = material('Chisel felt', [0.25, 0.27, 0.015] if color[0] > 0.8 and color[1] > 0.5
                           else [0.45, 0.025, 0.09], 0.85)
    vertices = [(x, y, z) for x in (0, 1) for y in (-0.0035, 0.0035)
                for z in (-0.0015, 0.0015)]
    vertices = [(-0.045 + 0.001 * (z / 0.0015) if x == 0 else -0.039, y, z)
                for x, y, z in vertices]
    nib = mesh('Chisel felt tip', vertices, BOX_FACES, bpy.data.objects['body'], tip_surface)
    collider(nib)
    label = material('Highlighter dark print', [0.12, 0.12, 0.10], 0.65)
    box('Flat face label', (0.027, 0.007, 0.00008), (0.014, 0, 0.00720),
        label, bpy.data.objects['body'], collision=False)


def white_capped(fine):
    if fine:
        geometry = {
            "segments": SEGMENTS, "seat_x_m": -0.0275, "barrel_end_x_m": 0.0795,
            "barrel_radius_m": 0.0048, "neck_length_m": 0.016,
            "neck_radius_m": 0.0038, "nib_length_m": 0.009,
            "nib_tip_radius_m": 0.0007, "nib_base_radius_m": 0.0026,
            "rear_start_x_m": 0.0785, "rear_end_x_m": 0.0825,
            "rear_radius_m": 0.0048, "cap_radius_m": 0.0055,
            "cap_length_m": 0.040, "end_thickness_m": 0.001,
            "lip_length_m": 0.002, "shell_inner_radius_m": 0.0047,
            "rigid_lip_clearance_m": 0.0003,
        }
    else:
        geometry = {
            "segments": SEGMENTS, "seat_x_m": -0.025, "barrel_end_x_m": 0.078,
            "barrel_radius_m": 0.0065, "neck_length_m": 0.019,
            "neck_radius_m": 0.0056, "nib_length_m": 0.007,
            "nib_tip_radius_m": 0.0015, "nib_base_radius_m": 0.003,
            "rear_start_x_m": 0.077, "rear_end_x_m": 0.080,
            "rear_radius_m": 0.0065, "cap_radius_m": 0.007,
            "cap_length_m": 0.045, "end_thickness_m": 0.001,
            "lip_length_m": 0.002, "shell_inner_radius_m": 0.0062,
            "rigid_lip_clearance_m": 0.0003,
        }
    profile = {
        "geometry": geometry, "barrel_color": [0.84, 0.85, 0.83],
        "cap_color": [0.88, 0.89, 0.86], "cap_transmission": 0,
        "rear_color": [0.84, 0.85, 0.83] if fine else [0.012, 0.015, 0.017],
        "rear_transmission": 0, "body_mass_kg": 0.007 if fine else 0.011,
        "cap_mass_kg": 0.002, "metal_point": fine,
    }
    if fine:
        profile.update({"clip_dimensions_m": [0.028, 0.0016, 0.0016],
                        "clip_position_m": [-0.023, 0, 0.0057],
                        "clip_color": [0.84, 0.85, 0.83]})
    capped_pen(profile)
    pen = bpy.data.objects['body']
    dark = material('Printed dark trim', [0.02, 0.025, 0.026], 0.55)
    if not fine:
        tube('Black shoulder band', 0.00655, 0.0064, 0.0017,
             (-0.0225, 0, 0), dark, pen, segments=SEGMENTS, collision=False)
        for i, (x, length) in enumerate(((-0.007, 0.009), (0.005, 0.005), (0.016, 0.006), (0.028, 0.004)), 1):
            label_patch(f'Printed barrel mark{i}', 0.00656, length, (x, 0, 0),
                        dark, pen, angle=math.pi / 2, span=0.60, segments=2)


def blue_click_pen():
    blue = material('Blue barrel plastic', [0.012, 0.055, 0.45], 0.28)
    black = material('Black grip and hardware', [0.009, 0.012, 0.016], 0.32)
    steel = material('Fine steel point', [0.50, 0.54, 0.57], 0.24, metallic=0.9)
    refill = material('Refill plastic', [0.31, 0.27, 0.18], 0.5)
    root = asset('pen', 'barrel midpoint; axis at origin',
                 '+X rear/button; -X writing tip; Z up', 'writing tip at -X')
    shell = body('pen.shell', root)
    tube('Blue barrel', 0.0048, 0.0028, 0.106, (0.010, 0, 0), blue, shell,
         segments=SEGMENTS, solid_collision=True)
    tube('Black tapered grip', 0.0009, 0.00065, 0.033, (-0.0595, 0, 0), black, shell,
         outer_end=0.0048, inner_end=0.0028, segments=SEGMENTS, solid_collision=True)
    tube('Black rear collar', 0.0050, 0.0035, 0.006, (0.066, 0, 0), black, shell,
         segments=SEGMENTS, solid_collision=True)
    box('Black pocket clip', (0.038, 0.0015, 0.0010), (0.040, 0, 0.00545),
        black, shell)
    box('Clip bridge', (0.004, 0.0025, 0.0015), (0.057, 0, 0.00525),
        black, shell)
    moving = body('pen.refill', root)
    cylinder('Black push button', 0.0026, 0.005, (0.0695, 0, 0), black, moving, segments=SEGMENTS)
    # Fully enclosed after the fixed shell/refill bodies are merged by physics.
    cylinder('Refill shaft', 0.0008, 0.131, (0.0005, 0, 0), refill, moving,
             segments=SEGMENTS, collision=False)
    cylinder('Fine ballpoint', 0.0003, 0.010, (-0.070, 0, 0), steel, moving,
             segments=SEGMENTS, collision=False)
    # Physics merges these bodies; the visual slider must not test their filled interiors.
    slider = joint('button_press', 'SLIDER', shell, moving, (0.067, 0, 0), (-0.003, 0))
    slider.rigid_body_constraint.disable_collisions = True
    organize_parts([
        ('pen.shell', 'barrel', ['Blue barrel']),
        ('pen.shell', 'grip', ['Black tapered grip']),
        ('pen.shell', 'collar', ['Black rear collar']),
        ('pen.shell', 'clip', ['Black pocket clip', 'Clip bridge']),
        ('pen.refill', 'button', ['Black push button']),
        ('pen.refill', 'shaft', ['Refill shaft']),
        ('pen.refill', 'tip', ['Fine ballpoint']),
    ])


def build_pen(style):
    builders = {
        'teal_marker': lambda: capped_pen(marker_profile([0.008, 0.07, 0.085])),
        'yellow_highlighter': lambda: highlighter([0.95, 0.85, 0.015]),
        'pink_highlighter': lambda: highlighter([0.95, 0.08, 0.29]),
        'blue_click': blue_click_pen,
        'white_fine': lambda: white_capped(True),
        'white_marker': lambda: white_capped(False),
    }
    builders[style]()
    if style != 'blue_click':
        organize_capped_parts()
