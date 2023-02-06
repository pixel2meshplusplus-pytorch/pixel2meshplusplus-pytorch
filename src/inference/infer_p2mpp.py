import torch
import numpy as np
import pickle
from pathlib import Path

from src.model.vgg16 import VGG16P2M
from src.model.multi_view_deformation_network import MultiViewDeformationNetwork
from src.data.shapenet import ShapeNetRenderings
from src.utils.tools import construct_feed_dict


class InferenceP2MPP:
    def __init__(self, dataset_type, path_to_perceptual_network_ckpt, path_to_mdn_ckpt):
        """
        :param dataset_type: one of: "test", "train", "validation" or "overfit"
        :param path_to_perceptual_network_ckpt: checkpoint path to weights of the trained perceptual network
        :param path_to_mdn_ckpt: checkpoint path to weights of the trained mdn network
        """
        self.dataset = ShapeNetRenderings(
            dataset_type,
            load_ground_truth=False,
            load_coarse_shapes=True
        )

        pkl = pickle.load(open('src/data/iccv_p2mpp.dat', 'rb'))
        feed_dict = construct_feed_dict(pkl)

        # define and load models
        self.perceptual_network = VGG16P2M(pretrained=True)
        self.mdn = MultiViewDeformationNetwork(339, feed_dict['sample_coord'], feed_dict['sample_adj'])

        # TODO: uncomment this once we pass correct checkpoints paths
        # self.perceptual_network.load_state_dict(torch.load(path_to_perceptual_network_ckpt, map_location='cpu'))
        # self.mdn.load_state_dict(torch.load(path_to_mdn_ckpt, map_location='cpu'))
        self.perceptual_network.eval()
        self.mdn.eval()

    def infer_single(self):
        # get random element in dataset
        idx =  torch.randint(len(self.dataset), (1,))[0]
        element = self.dataset[idx]

        # Forward pass
        with torch.no_grad():
            img_feat = self.perceptual_network(element['images'])["geometry_feature"]
            _, out2 = self.mdn(element["coarse_mesh"], element['cameras'], img_feat)


        # write output mesh result
        output_mesh = out2
        vert = np.hstack((np.full([output_mesh.shape[0], 1], 'v'), output_mesh))
        face = np.loadtxt('src/data/face3.obj', dtype='|S32')
        mesh = np.vstack((vert, face))

        pred_path = 'src/data/p2mpp_prediction.obj'
        np.savetxt(pred_path, mesh, fmt='%s', delimiter=' ')

        print('=> save to {}'.format(pred_path))
