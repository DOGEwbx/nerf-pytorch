import os
import torch
import numpy as np
import imageio 
import json
import torch.nn.functional as F
import cv2
from scipy.spatial.transform import Rotation as R

def load_ijrr_data(basedir, half_res=False, testskip=1):
    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        # with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
        #     metas[s] = json.load(fp)
        metas[s] = json.load(open(os.path.join(basedir, "event.json")))
    all_imgs = []
    all_poses = []
    counts = [0]
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        # if s=='train' or testskip==0:
        #     skip = 1
        # else:
        #     skip = testskip
            
        # for frame in meta['frames'][::skip]:
            # fname = os.path.join(basedir, frame['file_path'] + '.png')
            # imgs.append(imageio.imread(fname))
            # poses.append(np.array(frame['transform_matrix']))
        for pos in meta["poses"]:
            r = R.from_quat(pos[-4:])
            T = np.zeros((4,4),dtype = np.float32)
            T[:3,:3] = r.as_matrix()
            T[:3,3] = np.array(pos[1:4])
            T[3,3]= 1.0
            poses.append(T)
        imgs = np.load(meta["image_path"])
        if s =="val" or s=="test":
            imgs = imgs[:10]
            poses = poses[:10]
        # poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)
    
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]
    
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    
    H, W = imgs[0].shape[:2]
    # camera_angle_x = float(meta['camera_angle_x'])
    focal = 199#.5 * W / np.tan(.5 * camera_angle_x)
    
    render_poses = poses[0:10]#torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
    
    if half_res:
        assert True, "half_res not implemented"
        # H = H//2
        # W = W//2
        # focal = focal/2.

        # imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        # for i, img in enumerate(imgs):
        #     imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        # imgs = imgs_half_res
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()

        
    return imgs, poses, render_poses, [H, W, focal], i_split

