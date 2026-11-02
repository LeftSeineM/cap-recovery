"""White/black retractable pen: shell and one moving button/refill body.

Local +X points to the button; tip is -X; barrel center is Z=0.005 m.
The black rear housing is fixed; only its small end button translates.
"""

import bpy

from table_1000.modeling.asset_builders import asset, body, cylinder, generate, joint, material, tube, group_part


def build():
    white = material("White pen plastic", (0.78, 0.80, 0.78), 0.29)
    black = material("Black pen plastic", (0.009, 0.012, 0.014), 0.30)
    metal = material("Ballpoint steel", (0.5, 0.54, 0.57), 0.24, metallic=0.92)
    ink = material("Refill tube", (0.24, 0.20, 0.12), 0.35)
    root = asset("pen", "below barrel reference midpoint; barrel Z=0.005 m",
                 "+X length toward button; +Y width; +Z up", "ballpoint tip at -X")
    shell = body("White black pen shell", root)
    tube("White barrel", 0.0048, 0.0028, 0.102, (0.006, 0, 0.005), white, shell,
         solid_collision=True)
    tube("Black rear housing", 0.0051, 0.0032, 0.011, (0.0625, 0, 0.005), black, shell,
         solid_collision=True)
    tube("Black tapered nose", 0.0008, 0.0005, 0.042, (-0.066, 0, 0.005), black, shell,
         outer_end=0.0049, inner_end=0.0028, solid_collision=True)
    tube("Black shoulder", 0.0055, 0.0028, 0.003, (0.055, 0, 0.005), black, shell,
         collision=False)
    moving = body("White black pen button and refill", root)
    cylinder("Black push button", 0.003, 0.007, (0.0715, 0, 0.005), black, moving)
    # Both are fully inside the shell's outer envelope; physics merges these bodies.
    cylinder("Refill shaft", 0.0008, 0.144, (-0.004, 0, 0.005), ink, moving, collision=False)
    cylinder("Fine ballpoint", 0.0003, 0.010, (-0.0813, 0, 0.005), metal, moving, collision=False)
    # Physics merges these bodies; the visual slider must not test their filled interiors.
    slider = joint("button_press", "SLIDER", shell, moving, (0.068, 0, 0.005), (-0.003, 0))
    slider.rigid_body_constraint.disable_collisions = True

    organize_parts()


def organize_parts():
    """Group the original geometry while preserving every saved world pose."""
    owner = bpy.data.objects['White black pen shell']
    owner.name = 'pen.shell'
    for component, names in [
        ('barrel', ['White barrel']),
        ('rear_housing', ['Black rear housing']),
        ('nose', ['Black tapered nose']),
        ('shoulder', ['Black shoulder']),
    ]:
        visuals = [bpy.data.objects[name] for name in names]
        collisions = sorted([obj for obj in owner.children
                             if any(obj.name == "Collision_" + name
                                    or obj.name.startswith("Collision_" + name + "_")
                                    for name in names)], key=lambda obj: obj.name)
        group_part(owner.name + "." + component, owner, visuals, collisions)
    owner = bpy.data.objects['White black pen button and refill']
    owner.name = 'pen.refill'
    for component, names in [
        ('button', ['Black push button']),
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
