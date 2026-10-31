"""Read saved Blender geometry and native joints for the physical asset builder."""

import argparse
import json
from pathlib import Path
import sys


def describe(surface):
    shader = surface.node_tree.nodes.get('Principled BSDF')
    return {'name': surface.name,
            'color': list(shader.inputs['Base Color'].default_value[:3]),
            'roughness': float(shader.inputs['Roughness'].default_value),
            'metallic': float(shader.inputs['Metallic'].default_value),
            'transmission': float(shader.inputs['Transmission Weight'].default_value),
            'ior': float(shader.inputs['IOR'].default_value),
            'alpha': float(shader.inputs['Alpha'].default_value)}


def extract(blend):
    import bpy
    from table_1000.modeling.asset_tree import asset_forest
    from table_1000.modeling.geometry_checks import GeometryChecks
    bpy.ops.wm.open_mainfile(filepath=str(blend), load_ui=False)
    scene = bpy.context.scene
    scene.frame_set(0)
    if scene.rigidbody_world:
        scene.rigidbody_world.enabled = False
    bpy.context.view_layer.update()
    checks = GeometryChecks()
    if checks.deformables:
        raise ValueError('Stage-one deformable geometry has no physical export implementation')
    graph = bpy.context.evaluated_depsgraph_get()
    nodes = []

    def visit(branch):
        obj, children = branch
        world = obj.matrix_world
        local = obj.parent.matrix_world.inverted() @ world if obj.parent else world
        row = {'name': obj.name, 'parent': obj.parent.name if obj.parent else None,
               'matrix': [list(line) for line in local],
               'rigid': bool(obj.get('rigid_body_root')), 'part': bool(obj.get('semantic_part'))}
        role = obj.get('geometry_role')
        if role:
            evaluated = obj.evaluated_get(graph)
            mesh = evaluated.to_mesh()
            try:
                mesh.calc_loop_triangles()
                row.update(role=role, body=checks.owner[obj.name],
                           vertices=[list(v.co) for v in mesh.vertices],
                           faces=[list(face.vertices) for face in mesh.polygons],
                           triangles=[list(face.vertices) for face in mesh.loop_triangles])
                if role == 'collision':
                    row['shape'] = obj.rigid_body.collision_shape
                    if row['shape'] == 'SPHERE':
                        row['radius'] = checks.spheres[obj.name]
                else:
                    row['material'] = describe(mesh.materials[0])
                    if len(mesh.materials) > 1:     # per-face materials: every slot and each face's slot
                        row['materials'] = [describe(surface) for surface in mesh.materials]
                        row['face_materials'] = [face.material_index for face in mesh.polygons]
                    row['normals'] = [list(mesh.corner_normals[index].vector)
                                      for face in mesh.polygons for index in face.loop_indices]
            finally:
                evaluated.to_mesh_clear()
        nodes.append(row)
        for child in children:
            visit(child)

    for branch in asset_forest(scene.objects):
        visit(branch)
    joints = []
    for obj in scene.objects:
        joint = obj.rigid_body_constraint
        if joint and obj.get('preview_joint'):
            limits = ((joint.limit_lin_x_lower, joint.limit_lin_x_upper) if joint.type == 'SLIDER'
                      else (joint.limit_ang_z_lower, joint.limit_ang_z_upper))
            enabled = joint.use_limit_lin_x if joint.type == 'SLIDER' else joint.use_limit_ang_z
            joints.append({'name': obj.name, 'type': joint.type, 'body1': joint.object1.name,
                           'body2': joint.object2.name, 'matrix': [list(line) for line in obj.matrix_world],
                           'limits': list(limits) if enabled else None,
                           'disable_collisions': bool(joint.disable_collisions)})
    model = {'nodes': nodes, 'joints': joints, 'units': 'm', 'up_axis': 'Z'}
    # object.py may author this on bpy.context.scene; retain it in the
    # geometry description without changing rigid-body physics parameters.
    if 'max_drop_m' in scene:
        model['max_drop_m'] = float(scene['max_drop_m'])
    return model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--blend', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(extract(args.blend.resolve())) + '\n')


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    main(sys.argv[sys.argv.index('--') + 1:])
