"""Run MPlib CLIK on the target's ancestor chain, retaining full-model sampling."""

import xml.etree.ElementTree as ET

import numpy as np


class ChainKinematics:
    """Delegate FK and random restarts to the full model; shrink only CLIK work.

    Branches outside the TCP ancestry have identically zero Jacobian columns.
    Their configurations, and the full model's random stream, stay unchanged.
    Collision checking continues to use the original whole-body planner.
    """

    def __init__(self, full, urdf, tcp):
        from mplib.kinematics.pinocchio import PinocchioModel
        self.full = full
        root = ET.parse(urdf).getroot()
        parents = {joint.find('child').get('link'): joint for joint in root.findall('joint')}
        links, link = {tcp}, tcp
        while link in parents:
            link = parents[link].find('parent').get('link')
            links.add(link)
        for element in list(root):
            if (element.tag == 'link' and element.get('name') not in links
                    or element.tag == 'joint' and element.find('child').get('link') not in links):
                root.remove(element)
        for element in root.findall('link'):
            for shape in element.findall('visual') + element.findall('collision'):
                element.remove(shape)
        self.chain = PinocchioModel.create_from_urdf_string(ET.tostring(root, encoding='unicode'))
        names = [name for name in full.get_joint_names() if name in self.chain.get_joint_names()]
        self.chain.set_joint_order(names)
        self.indices = np.array([full.get_joint_names().index(name) for name in names])
        self.full_link = full.get_link_names().index(tcp)
        self.chain_link = self.chain.get_link_names().index(tcp)

    def __getattr__(self, name):
        return getattr(self.full, name)

    def compute_IK_CLIK(self, index, target, q_init, mask=(), **kwargs):
        if index != self.full_link:
            return self.full.compute_IK_CLIK(index, target, q_init, mask, **kwargs)
        q_init = np.asarray(q_init)
        selected = np.asarray(mask)[self.indices] if len(mask) else []
        q, success, error = self.chain.compute_IK_CLIK(
            self.chain_link, target, q_init[self.indices], selected, **kwargs)
        result = q_init.copy()
        result[self.indices] = q
        return result, success, error
