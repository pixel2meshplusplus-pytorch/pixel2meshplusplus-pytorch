import torch
import numpy as np
import pickle
from pathlib import Path

from src.model.multi_view_p2m import MVP2M
from src.model.vgg16 import VGG16P2M
from src.utils.loss.losses import P2MLoss
from src.data.shapenet import ShapeNetRenderings
from src.utils.tools import construct_feed_dict


class InferenceMVP2M:
    def __init__(self, dataset_type, path_to_ckpt):
        """
        :param dataset_type: one of: "test", "train", "validation" or "overfit"
        :param path_to_perceptual_network_ckpt: checkpoint path to weights of the trained perceptual network
        :param path_to_mvp2m_ckpt: checkpoint path to weights of the trained mvp2m network
        """
        self.dataset = ShapeNetRenderings(dataset_type)

        pkl = pickle.load(open('src/data/iccv_p2mpp.dat', 'rb'))
        feed_dict = construct_feed_dict(pkl)

        # define and load models
        self.perceptual_network = VGG16P2M(pretrained=True).cuda()
        self.mvp2m = MVP2M(feed_dict['supports'], feed_dict['pool_idx'], 2883, 192, 3).cuda()
        
        checkpoint = torch.load(path_to_ckpt)              
        self.perceptual_network.load_state_dict(checkpoint['perceptual_network'])
        self.mvp2m.load_state_dict(checkpoint['mvp2m'])
        
#         self.perceptual_network.load_state_dict(torch.load(path_to_perceptual_network_ckpt, map_location='cpu'))
#         self.mvp2m.load_state_dict(torch.load(path_to_mvp2m_ckpt, map_location='cpu'))
        self.perceptual_network.eval()
        self.mvp2m.eval()

        self.ellipsoid = feed_dict['ellipsoid_feature_X'].cuda()
        # self.lape_idx = feed_dict['lape_idx']
        # self.edges = feed_dict['edges']
        # self.faces = feed_dict['faces_triangle']
        self.lape_idx = [torch.as_tensor(l, dtype=torch.long).cuda() for l in feed_dict['lape_idx']]
        self.edges = [torch.as_tensor(e, dtype=torch.long).cuda() for e in feed_dict['edges']]
        self.faces = [torch.as_tensor(f, dtype=torch.long).cuda() for f in feed_dict['faces_triangle']]

    def infer_single(self):
        # get random element in dataset
        idx =  torch.randint(len(self.dataset), (1,))[0]
        element = self.dataset[idx]

        img_all_view = element['images'].cuda()
        cameras = element['cameras'].cuda()

        # Forward pass
        with torch.no_grad():
            img_feat = self.perceptual_network(img_all_view)["semantic_feature"]
            output = self.mvp2m(self.ellipsoid, img_feat, cameras)
            
        # loss_func = P2MLoss(self.lape_idx, self.edges, self.faces).cuda()
        # loss = loss_func(output, element)[0]
        
        # print("Loss: ", loss)
        
        
        # write output mesh result
        output_mesh = output["pred_coord"][2].squeeze().cpu()
        vert = np.hstack((np.full([output_mesh.shape[0], 1], 'v'), output_mesh))
        face = np.loadtxt('src/data/face3.obj', dtype='|S32')
        mesh = np.vstack((vert, face))

        pred_path = 'src/data/mvp2m_prediction.obj'
        np.savetxt(pred_path, mesh, fmt='%s', delimiter=' ')

        print('=> save to {}'.format(pred_path))


    def infer_all(self):
        # Create folder for saving coarse shapes
        course_shapes_dir = f'src/coarse_shapes'
        Path(course_shapes_dir).mkdir(exist_ok=True, parents=True)

        for idx in range(len(self.dataset)):
            element = self.dataset[idx]

            img_all_view = element['images']
            cameras = element['cameras']
            category_id = element['category_id']
            item_id = element['item_id']

            # Forward pass
            with torch.no_grad():
                img_feat = self.perceptual_network(img_all_view)["semantic_feature"]
                output = self.mvp2m(self.ellipsoid, img_feat, cameras)


            # write output mesh result
            output_mesh = output["pred_coord"][2].squeeze()

            pred_path = course_shapes_dir + "/" + category_id + "_" + item_id + "_predict.xyz"
            np.savetxt(pred_path, output_mesh)

        print('=> save to folder {}'.format(course_shapes_dir))
        