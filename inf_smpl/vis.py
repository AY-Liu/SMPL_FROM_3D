import os
os.environ["PYOPENGL_PLATFORM"]="osmesa"
os.environ["MUJOCO_GL"]="osmesa"


import numpy as np
import cv2
import pyrender
import trimesh
import matplotlib.pyplot as plt
from pyrender.node import Node
from pyrender.light import DirectionalLight
from math import sqrt


DEFAULT_SCENE_SCALE=2


def create_raymond_lights():
    thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])

    nodes = []

    for phi, theta in zip(phis, thetas):
        xp = np.sin(theta) * np.cos(phi)
        yp = np.sin(theta) * np.sin(phi)
        zp = np.cos(theta)

        z = np.array([xp, yp, zp])
        z = z / np.linalg.norm(z)
        x = np.array([-z[1], z[0], 0.0])
        if np.linalg.norm(x) == 0:
            x = np.array([1.0, 0.0, 0.0])
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)

        matrix = np.eye(4)
        matrix[:3, :3] = np.c_[x, y, z]
        nodes.append(Node(
            light=DirectionalLight(color=np.ones(3), intensity=1.0),
            matrix=matrix
        ))

    return nodes


def compute_initial_camera_pose(scene):
    centroid = scene.centroid
    scale = scene.scale
    if scale == 0.0:
        scale = DEFAULT_SCENE_SCALE

    s2 = 1.0 / np.sqrt(2.0)
    cp = np.eye(4)
    cp[:3, :3] = np.array([
        [0.0, -s2, s2],
        [1.0, 0.0, 0.0],
        [0.0, s2, s2]
    ])
    rotation=np.array([
        [0,1,0],
        [-1,0,0],
        [0,0,1],
    ])
    cp[:3,:3]=np.dot(rotation,cp[:3,:3])
    rotation=np.array([
        [1,0,0],
        [0,1/2,sqrt(3)/2],
        [0,-sqrt(3)/2,1/2],
    ])
    cp[:3,:3]=np.dot(rotation,cp[:3,:3])
    hfov = np.pi / 6.0
    dist = scale / (2.0 * np.tan(hfov))
    cp[:3, 3] = dist * np.array([1.0, 0.0, 1.0]) + centroid+np.array([-1.5,0.7,0])

    return cp


def render_smpl(verts,faces):
    mesh=trimesh.Trimesh(verts,faces)
    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0,
        alphaMode='OPAQUE',baseColorFactor=(1., 1., 1.))
    mesh = pyrender.Mesh.from_trimesh(mesh,material=material)
    scene = pyrender.Scene()
    scene.add(mesh)
    zfar = max(scene.scale * 10.0, 100)
    znear = min(scene.scale / 10.0, 0.05)
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 2.5, znear=znear, zfar=zfar)
    default_camera_pose=compute_initial_camera_pose(scene)
    # print(default_camera_pose)
    scene.add(camera, pose=default_camera_pose)
    raymond_lights=create_raymond_lights()
    camera_node = Node(
        matrix=default_camera_pose, camera=camera
    )
    scene.add_node(camera_node)
    for n in raymond_lights:
        n.light.intensity = 1
        if not scene.has_node(n):
            scene.add_node(n,parent_node=camera_node)
    r = pyrender.OffscreenRenderer(512, 512)
    img, _ = r.render(scene)
    plt.figure()
    plt.axis('off')
    plt.imshow(img)
    plt.show()
    return img

def render_smpl_demo(verts,faces,camera_pose,img=None,add_back=False,visual=False):
    mesh = trimesh.Trimesh(verts, faces)
    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.1,
        alphaMode='OPAQUE',baseColorFactor=(1.,1., 1.))  # (122/255., 192/255., 245/255.),(91/255.,174/255, 35/255)
    mesh = pyrender.Mesh.from_trimesh(mesh,material=material)
    scene = pyrender.Scene(bg_color=(0,0,0))
    scene.add(mesh)
    camera_pose=camera_pose
    scale=np.diag([1,1,1,1])
    l2r=np.diag([1,1,-1,1])
    camera_pose=np.dot(l2r,camera_pose)
    camera_pose=np.dot(scale,camera_pose)
    camera = pyrender.PerspectiveCamera(yfov=44.10*np.pi/180) #44.10,58.99,66.31
    camera_node = Node(
        matrix=camera_pose, camera=camera
    )
    scene.add_node(camera_node)
    raymond_lights=create_raymond_lights()
    for n in raymond_lights:
        n.light.intensity = 1.5
        if not scene.has_node(n):
            scene.add_node(n,parent_node=camera_node)
    r = pyrender.OffscreenRenderer(854,480)
    # r = pyrender.OffscreenRenderer(1024, 1024)
    smpl_img, smpl_depth = r.render(scene)
    if add_back:
        # if smpl_img.shape[2] == 3:  # fail to generate transparent channel
        #     valid_mask = (smpl_depth > 0)[:, :, None]
        #     smpl_img = np.dstack((smpl_img, (valid_mask * 255).astype(np.uint8)))
        # smpl_img = smpl_img[..., [2, 1, 0, 3]]
        # rend_cat=cv2.addWeighted(
        #     cv2.bitwise_and(img, 255 - smpl_img[:, :, 3:4].repeat(3, 2)), 1,
        #     cv2.bitwise_and(smpl_img[:, :, :3], smpl_img[:, :, 3:4].repeat(3, 2)), 1, 0
        # )
        smpl_img = smpl_img.astype(np.float32)
        valid_mask = (smpl_depth > 0)[:, :, None]
        rend_cat = (smpl_img[:, :, :3] * valid_mask +
                        (1 - valid_mask) * img).astype(np.uint8)
    else:
        rend_cat=smpl_img
    if visual:
        plt.figure()
        plt.axis('off')
        plt.imshow(rend_cat,aspect="equal")
        plt.show()
        plt.close()
    return rend_cat


if __name__ == '__main__':
    img = cv2.imread("/mnt/data/images/M001P001A001R001/000001.jpg")
    faces = np.load("/root/smpl/base/faces.npy")
    verts = np.load("/root/smpl/base/vertices.npy")
    camera_pose = np.array([[-0.9999773, -0.00404104, -0.00539746,  35.53189],
                                    [-0.004719454, 0.9912043, 0.1322568, 2.188294],
                                    [-0.00481553, -0.1322793, 0.9912007, 21.58087],
                                    [0.00000, 0.00000, 0.00000, 1.00000]])
    render_smpl_demo(verts, faces,camera_pose=camera_pose,img=img,add_back=True,visual=True)
    # render_smpl(verts,faces)