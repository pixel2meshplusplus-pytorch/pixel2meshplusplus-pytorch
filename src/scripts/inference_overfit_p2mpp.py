# To run this file:
# 1. Go to main directory: pixel2meshplusplus-pytorch
# 2. Run: python inference_overfit_p2mpp.py

from src.inference.infer_p2mpp import InferenceP2MPP

def main():
    inference = InferenceP2MPP(
        dataset_type='overfit',
        path_to_perceptual_network_ckpt='src/runs/p2mpp_overfitting/perceptual_network_best.ckpt',
        path_to_p2mpp_ckpt='src/runs/p2mpp_overfitting/model_best.ckpt'
    )

    inference.infer_single()

if __name__ == '__main__':
    main()
