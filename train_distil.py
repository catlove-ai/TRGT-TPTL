import logging
import os.path

from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from config.features import ALL_FEATURES, ACTIVE_FEATURES
from modeling.base_model import BaseModel
from modeling.distill_model import DistillModel
from utils.common import get_root_path
from utils.datasets import BaseDataset, base_collate, MultiLabelDataset, multi_label_collate
from utils.run import EarlyStopping, Trainer, set_seed

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(filename)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()]
                    )


def train(jf_game_id, student_model, train_start_date, train_end_date, val_start_date, val_end_date, save_path, teacher_weights_path, features=ALL_FEATURES, device='cuda:0', max_row=None, minus_label_col=None):
    # args
    epochs = 10
    batch_size = 1024
    lr = 0.005
    lr_scheduler_patience = 0
    early_stop_patience = 2

    # log feature info
    feature_number = {
        'dense': len(features['dense']),
        'sparse': len(features['sparse']),
        'seq': len(features['seq'])
    }
    logging.info(f'feature number:{feature_number}')

    # model
    # student_model = 'TRGT'
    model = DistillModel(teacher_weights_path=teacher_weights_path, student_model=student_model)

    # optim
    optimizer = optim.Adam(model.parameters(), lr)
    # scheduler = ReduceLROnPlateau(optimizer, patience=lr_scheduler_patience, min_lr=1e-6)
    scheduler = None
    early_stopping = EarlyStopping(patience=early_stop_patience)

    BaseDataset_clss = None
    co_fn = None
    if student_model == 'TRGT':
        BaseDataset_clss = MultiLabelDataset
        co_fn = multi_label_collate
    elif student_model == 'DNN':
        BaseDataset_clss = BaseDataset
        co_fn = base_collate
    # data
    train_dataset = BaseDataset_clss(jf_game_id, train_start_date, train_end_date, features=features, max_rows=max_row, minus_label_col=minus_label_col)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=co_fn)
    val_dataset = BaseDataset_clss(jf_game_id, val_start_date, val_end_date, features=features, max_rows=max_row, minus_label_col=minus_label_col)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=co_fn)

    # train
    trainer = Trainer(model, optimizer, scheduler, early_stopping)
    trainer.train(save_path, epochs, train_loader, val_loader)


if __name__ == '__main__':
    set_seed(42)
    for jf_game_id in ['wxsh']:
        save_path = os.path.join(get_root_path(), 'checkpoints/wxsh', f'{jf_game_id}_distill_dnn_active_th_seed_42')
        teacher_weights_path = r'D:\codes\www\checkpoints\train_model\wxsh_DNN_1_layer_6h_tc_seed_42\10.pth'
        train(jf_game_id, 'DNN', 805, 825, 826, 829,
              save_path=save_path, features=ALL_FEATURES, minus_label_col=None, teacher_weights_path=teacher_weights_path, max_row=None)

    for jf_game_id in ['wxsh']:
        save_path = os.path.join(get_root_path(), 'checkpoints/wxsh', f'{jf_game_id}_distill_trgt_active_th_seed_42')
        teacher_weights_path = r'D:\codes\www\checkpoints\train_model\wxsh_DNN_1_layer_6h_tc_seed_42\10.pth'
        train(jf_game_id,'TRGT', 805, 825, 826, 829,
              save_path=save_path, features=ALL_FEATURES, minus_label_col=None, teacher_weights_path=teacher_weights_path, max_row=None)

