from src.training.train_multi_view_pixel2mesh import main

config = {
    'experiment_name': 'mvp2m_overfitting',
    'device': 'cuda:0',  # change this to cpu if you do not have a GPU
    'is_overfit': True,
    'resume_ckpt': None,
    # 'resume_ckpt': 'src/runs/mvp2m_overfitting/checkpoint.ckpt',
    'learning_rate': 1e-3, # from Pixel2Mesh++ paper
    'weight_decay': 0, # from Pixel2Mesh++ paper
    'max_epochs': 100,
    'print_every_n': 10,
    'validate_every_n': 25,
    'send_telegram_message': True,
}

main(config)
