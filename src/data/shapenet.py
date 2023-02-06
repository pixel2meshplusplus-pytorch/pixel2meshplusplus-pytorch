import os
import cv2

import torch
import numpy as np
import pickle

class ShapeNetRenderings:
    def __init__(self, data_type="train", load_ground_truth=True, load_coarse_shapes=False): # data_type should be either "test", "train", "validation" or "overfit"
        self.load_ground_truth = load_ground_truth
        self.load_coarse_shapes = load_coarse_shapes
        self.data_type = data_type
        self.data_root = os.path.join(os.getcwd(), "src", "data")
        self.shapenet_data_root = os.path.join(self.data_root, "shapenet")

        txt_name = data_type + "_list.txt"
        self.file_name = os.path.join(self.data_root, txt_name)
        self.item_names : list[str] = []
        with open(self.file_name, "r") as file:
            while True:
                line = file.readline().strip()
                if not line:
                    break
                self.item_names.append(line)

    def __getitem__(self, index):
        path = self.item_names[index]

        ids = path.split('_')
        category = ids[0]
        item_id = ids[1]
        img_path = os.path.join(self.shapenet_data_root, "images", category, item_id, 'rendering')
        camera_meta_data = np.loadtxt(os.path.join(img_path, 'rendering_metadata.txt'))

        imgs = np.zeros((3, 224, 224, 3))
        poses = np.zeros((3, 5))
        for idx, view in enumerate([0, 6, 7]):
            img = cv2.imread(os.path.join(img_path, str(view).zfill(2) + '.png'), cv2.IMREAD_UNCHANGED)
            img[np.where(img[:, :, 3] == 0)] = 255
            img = cv2.resize(img, (224, 224))
            img_inp = img.astype('float16') / 255.0
            imgs[idx] = img_inp[:, :, :3]
            poses[idx] = camera_meta_data[view]
            
        if self.load_ground_truth:
            with open(os.path.join(self.shapenet_data_root, "ground_truth", self.data_type, path), 'rb') as pickle_file:
                pkl = pickle.load(pickle_file, encoding='bytes')
                points_number, _ = pkl.shape

                points = pkl[:,:3] # get the first 3 element of every list
                normals = pkl[:,3:6] # get the last 3 element of every list

        if self.load_coarse_shapes:
            #mesh = np.loadtxt(os.path.join(os.getcwd(), "coarse_shapes", category + '_' + item_id + '_predict.xyz'))
            mesh= torch.tensor(np.loadtxt('src/data/coarse.xyz'), dtype=torch.float32)

        return {
            "category_id": category,
            "item_id": item_id,
            "images": torch.tensor(imgs, dtype=torch.float32).permute(0, 3, 1, 2),
            "cameras": torch.tensor(poses, dtype=torch.float32),
            "points": torch.tensor(points, dtype=torch.float32).unsqueeze(0) if self.load_ground_truth else None,
            "normals": torch.tensor(normals, dtype=torch.float32).unsqueeze(0) if self.load_ground_truth else None,
            "coarse_mesh": self.load_coarse_shapes and mesh 
        }

    def __len__(self):
        return len(self.item_names)
