from src.training.train_multi_view_pixel2mesh import main

config = {
    'experiment_name': 'mvp2m_generalization',
    'device': 'cuda:0',  # change this to cpu if you do not have a GPU
    'is_overfit': False,
    'resume_mvp2m_perceptual_network_ckpt': None,
    'resume_mvp2m_ckpt': None,
    'learning_rate': 1e-5, # from Pixel2Mesh++ paper
    'weight_decay': 1e-5, # from Pixel2Mesh++ paper
    'max_epochs': 30, # from Pixel2Mesh++ paper
    'print_every_n': 10,
    'validate_every_n': 1000,
    'send_telegram_message': True,
}

main(config)