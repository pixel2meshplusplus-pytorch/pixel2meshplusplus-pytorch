from src.training.train_pixel2meshplusplus import main

config = {
    'experiment_name': 'p2mpp_overfitting',
    'device': 'cuda:0',  # change this to cpu if you do not have a GPU
    'is_overfit': True,
    'resume_ckpt': None,
    # 'resume_ckpt': 'src/runs/p2mpp_overfitting/checkpoint.ckpt',
    'learning_rate': 1e-4, # from Pixel2Mesh++ paper
    'weight_decay': 0, # from Pixel2Mesh++ paper
    'max_epochs': 100,
    'print_every_n': 10,
    'validate_every_n': 25,
    'send_telegram_message': True,
}

main(config)