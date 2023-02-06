from src.training.train_pixel2meshplusplus import main

config = {
    'experiment_name': 'p2mpp_generalization',
    'device': 'cuda:0',  # change this to cpu if you do not have a GPU
    'is_overfit': False,
    'resume_p2mpp_perceptual_network_ckpt': None,
    'resume_p2mpp_ckpt': None,
    'learning_rate': 1e-6, # from Pixel2Mesh++ paper
    'weight_decay': 1e-5, # from Pixel2Mesh++ paper
    'max_epochs': 20, # from Pixel2Mesh++ paper
    'print_every_n': 10,
    'validate_every_n': 1000,
}

main(config)