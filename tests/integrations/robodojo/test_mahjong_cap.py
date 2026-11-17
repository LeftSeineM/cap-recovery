import hashlib,json
from pathlib import Path
import numpy as np
from table_1000.integrations.robodojo import mahjong_cap as cap
from table_1000.integrations.robodojo.mahjong_cap.style import template,angle

ROOT=Path(cap.__file__).parent

import unittest

class TestMahjongCap(unittest.TestCase):
    def test_published_policy_and_short_priors_match_native_freeze(self):
        manifest=json.loads((ROOT/'release_manifest.json').read_text())
        for name,digest in manifest['policy_and_short_prior_files_sha256'].items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest

    def test_low_velocity_is_required_even_when_pose_is_upright(self):
        q=np.array([1.,1.,0.,0.])/np.sqrt(2)
        poses={'mahjong3_0':np.r_[np.zeros(3),q]}
        assert cap.entry_targets_settled(poses,{'mahjong3_0':np.zeros(6)},poses)
        assert not cap.entry_targets_settled(poses,{'mahjong3_0':np.r_[np.zeros(3),.21,0,0]},poses)

    def test_face_down_and_partial_entry_need_a_new_capability(self):
        poses={'mahjong3_0':np.r_[np.zeros(3),[0.,1.,0.,0.]]}
        assert angle(poses['mahjong3_0'][3:],[0,0,1])>179
        assert not cap.entry_targets_resolved(poses,poses)
        theta=np.radians(50)/2
        poses['mahjong3_0'][3:]=[np.cos(theta),np.sin(theta),0,0]
        assert not cap.entry_targets_resolved(poses,poses)

    def test_contact_template_translation_preserves_26_waypoints(self):
        with np.load(ROOT/'validated_push_style.npz') as z:
            source=np.asarray(z['source_object_pose'],dtype=float).reshape(7)
        first,pre,forward=template(source)
        delta=np.array([.05,-.03,.01]);shifted=source.copy();shifted[:3]+=delta
        second,pre2,forward2=template(shifted)
        assert first.shape==second.shape==(26,7)
        np.testing.assert_allclose(second[:,:3],first[:,:3]+delta,atol=1e-12)
        np.testing.assert_allclose(second[:,3:],first[:,3:],atol=1e-12)
        np.testing.assert_allclose(pre2[:3],pre[:3]+delta,atol=1e-12)
        np.testing.assert_allclose(forward2,forward,atol=1e-12)
