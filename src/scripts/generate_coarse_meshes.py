# To run this file:
# 1. Go to main directory: pixel2meshplusplus-pytorch
# 2. Run: python generate_coarse_meshes.py

from src.inference.infer_mvp2m import InferenceMVP2M

def main():
    inference = InferenceMVP2M(
        dataset_type='train',
        path_to_perceptual_network_ckpt='src/runs/mvp2m_training/perceptual_network_best.ckpt',
        path_to_mvp2m_ckpt='src/runs/mvp2m_training/model_best.ckpt'
    )

    inference.infer_all()

if __name__ == '__main__':
    main()
