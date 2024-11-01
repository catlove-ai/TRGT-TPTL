import logging
import os.path

from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from config.features import ALL_FEATURES, ACTIVE_FEATURES
from modeling.pretrain_model import PretrainedModel
from utils.common import get_root_path
from utils.datasets import PretrainDataset, pretrain_collate
from utils.run import EarlyStopping, Trainer, set_seed

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(filename)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()]
                    )


def pretrain(jf_game_id, train_start_date, train_end_date, val_start_date, val_end_date, save_path, device='cuda:0', max_row=None):
    # args
    epochs = 20
    batch_size = 1024
    lr = 0.001
    lr_scheduler_patience = 0
    early_stop_patience = 2

    # model
    pretrain_model = PretrainedModel(device=device)

    # optim
    optimizer = optim.Adam(pretrain_model.parameters(), lr)
    # scheduler = ReduceLROnPlateau(optimizer, patience=lr_scheduler_patience, min_lr=1e-6)
    scheduler = None
    early_stopping = EarlyStopping(patience=early_stop_patience)

    # data
    train_dataset = PretrainDataset(jf_game_id, train_start_date, train_end_date, max_rows=max_row)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pretrain_collate)

    # train
    trainer = Trainer(pretrain_model, optimizer, scheduler)
    trainer.train(save_path, epochs, train_loader)


if __name__ == '__main__':
    set_seed(42)
    for jf_game_id in ['dhp']:
            save_path = os.path.join(get_root_path(), 'checkpoints/pretrain_model', f'{jf_game_id}_pretrain_810~825_th_seed_42')
            pretrain(jf_game_id, 815, 825, 826, 826, save_path=save_path, max_row=None)

