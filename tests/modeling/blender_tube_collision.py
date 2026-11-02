"""Exercise real Blender meshes for the optional solid collision envelope."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import bpy
import bmesh
from table_1000.modeling.asset_builders import asset, body, material, tube

root = asset('test', 'origin', 'XYZ', '+X')
surface = material('test', (.2, .3, .4))
for axis in ('X', 'Y', 'Z'):
    meshes, proxies = [], []
    for solid in (False, True):
        owner = body(f'{axis}-{solid}', root)
        visual = tube(f'tube-{axis}-{solid}', .006, .003, .10, (.01, .02, .03), surface,
                      owner, axis=axis, outer_end=.004, inner_end=.002,
                      solid_collision=solid)
        meshes.append(visual.data)
        proxies.append([child for child in owner.children if child.get('geometry_role') == 'collision'])
    assert [tuple(v.co) for v in meshes[0].vertices] == [tuple(v.co) for v in meshes[1].vertices]
    assert [tuple(f.vertices) for f in meshes[0].polygons] == [tuple(f.vertices) for f in meshes[1].polygons]
    assert len(proxies[0]) == 16 and len(proxies[1]) == 1
    proxy = proxies[1][0]
    assert proxy.rigid_body.collision_shape == 'CONVEX_HULL'
    assert {tuple(v.co) for v in proxy.data.vertices} == {tuple(v.co) for v in meshes[0].vertices[:32]}
    bm = bmesh.new()
    bm.from_mesh(proxy.data)
    assert all(edge.is_manifold for edge in bm.edges)
    assert bm.calc_volume() > 0.
    bm.free()
owner = body('visual-only', root)
tube('visual-only', .006, .003, .1, (0., 0., 0.), surface, owner,
     collision=False, solid_collision=True)
assert not any(child.get('geometry_role') == 'collision' for child in owner.children)
print('TUBE_COLLISION_CHECK_COMPLETED', flush=True)
