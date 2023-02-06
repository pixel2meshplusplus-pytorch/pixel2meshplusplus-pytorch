# To run this file:
# 1. Go to main directory: pixel2meshplusplus-pytorch
# 2. Run: python inference_overfit_mvp2m.py

from src.inference.infer_mvp2m import InferenceMVP2M

def main():
    inference = InferenceMVP2M(
        dataset_type='overfit',
        path_to_ckpt='src/runs/mvp2m_overfitting/checkpoint.ckpt'
    )

    inference.infer_single()

if __name__ == '__main__':
    main()
