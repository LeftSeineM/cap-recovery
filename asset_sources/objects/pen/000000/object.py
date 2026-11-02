"""Black retractable ballpoint: hollow shell and one moving button/refill body.

Local +X points to the button; tip is -X; barrel center is Z=0.006 m.
One native slider describes travel, without spring or click-lock dynamics.
"""

import bpy

from table_1000.modeling.asset_builders import asset, body, box, cylinder, generate, joint, material, tube, group_part


def build():
    black = material("Black pen lacquer", (0.007, 0.010, 0.012), 0.19)
    metal = material("Brushed nickel hardware", (0.50, 0.54, 0.57), 0.24, metallic=0.92)
    ink = material("Refill tube", (0.24, 0.20, 0.12), 0.35)
    root = asset("pen", "below barrel midpoint; barrel Z=0.006 m",
                 "+X length toward button; +Y width; +Z up", "ballpoint tip at -X")
    shell = body("Black pen shell", root)
    tube("Black lacquer barrel", 0.0056, 0.0034, 0.113, (0, 0, 0.006), black, shell,
         solid_collision=True)
    tube("Hollow metal nose", 0.0011, 0.0008, 0.024, (-0.0685, 0, 0.006), metal, shell,
         outer_end=0.0056, inner_end=0.0034, solid_collision=True)
    for x in (-0.010, -0.005, 0, 0.005):
        tube("Silver grip band", 0.0058, 0.00555, 0.0017, (x, 0, 0.006), metal, shell,
             collision=False)
    box("Pocket clip", (0.045, 0.0024, 0.0025), (0.031, 0.005, 0.009), black, shell,
        bevel=0.0008)
    box("Pocket clip bridge", (0.004, 0.0055, 0.003), (0.050, 0.0025, 0.011), black, shell,
        bevel=0.0008)
    moving = body("Black pen button and refill", root)
    cylinder("Silver push button", 0.003, 0.012, (0.062, 0, 0.006), metal, moving)
    # Both are fully inside the shell's outer envelope; physics merges these bodies.
    cylinder("Refill shaft", 0.0009, 0.127, (-0.0075, 0, 0.006), ink, moving, collision=False)
    cylinder("Fine ballpoint", 0.00045, 0.010, (-0.0753, 0, 0.006), metal, moving, collision=False)
    # Physics merges these bodies; the visual slider must not test their filled interiors.
    slider = joint("button_press", "SLIDER", shell, moving, (0.056, 0, 0.006), (-0.003, 0))
    slider.rigid_body_constraint.disable_collisions = True

    organize_parts()


def organize_parts():
    """Group the original geometry while preserving every saved world pose."""
    owner = bpy.data.objects['Black pen shell']
    owner.name = 'pen.shell'
    for component, names in [
        ('barrel', ['Black lacquer barrel']),
        ('nose', ['Hollow metal nose']),
        ('clip', ['Pocket clip']),
        ('clip_bridge', ['Pocket clip bridge']),
        ('grip_band1', ['Silver grip band']),
        ('grip_band2', ['Silver grip band.001']),
        ('grip_band3', ['Silver grip band.002']),
        ('grip_band4', ['Silver grip band.003']),
    ]:
        visuals = [bpy.data.objects[name] for name in names]
        collisions = sorted([obj for obj in owner.children
                             if any(obj.name == "Collision_" + name
                                    or obj.name.startswith("Collision_" + name + "_")
                                    for name in names)], key=lambda obj: obj.name)
        group_part(owner.name + "." + component, owner, visuals, collisions)
    owner = bpy.data.objects['Black pen button and refill']
    owner.name = 'pen.refill'
    for component, names in [
        ('button', ['Silver push button']),
        ('shaft', ['Refill shaft']),
        ('tip', ['Fine ballpoint']),
    ]:
        visuals = [bpy.data.objects[name] for name in names]
        collisions = sorted([obj for obj in owner.children
                             if any(obj.name == "Collision_" + name
                                    or obj.name.startswith("Collision_" + name + "_")
                                    for name in names)], key=lambda obj: obj.name)
        group_part(owner.name + "." + component, owner, visuals, collisions)


def main(argv=None):
    generate(build, __file__, argv)


if __name__ == "__main__":
    main()
