from base.experiment import GenericExperiment
from base.utils import load_pickle, ensure_dir
from base.loss_function import CCCLoss
from base.trainer import Trainer
from base.parameter_control import ResnetParamControl
from model.model import my_temporal, SA_TCN, MSA_TCN
from base.dataset_old_backup import DataArranger, MyDatasetPreLoad
from base.checkpointer import Checkpointer

import os

import numpy as np


class Experiment(GenericExperiment):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.task = args.task
        self.case = args.case

        # For tcn and lstm on regression tasks.
        self.bandpower_dim = args.bandpower_dim
        self.cnn1d_embedding_dim = args.cnn1d_embedding_dim
        self.cnn1d_channels = args.cnn1d_channels
        self.cnn1d_kernel_size = args.cnn1d_kernel_size
        self.cnn1d_dropout = args.cnn1d_dropout
        self.lstm_embedding_dim = args.lstm_embedding_dim
        self.lstm_hidden_dim = args.lstm_hidden_dim
        self.lstm_dropout = args.lstm_dropout
        self.num_eeg_chan = args.num_eeg_chan
        self.num_f = args.num_f

        # For parameter control.
        self.release_count = args.release_count
        self.gradual_release = args.gradual_release
        self.backbone_mode = args.backbone_mode

        self.num_folds = args.num_folds

        # 如果设定为跑所有折，则根据实际的 num_folds 生成对应的列表 (比如 5 折就是 [0, 1, 2, 3, 4])
        if self.folds_to_run[0] == "all":
            self.folds_to_run = np.arange(self.num_folds)

    def prepare(self):
        self.config = self.get_config()

        self.feature_dimension = self.get_feature_dimension(self.config)
        self.multiplier = self.get_multiplier(self.config)
        self.time_delay = self.get_time_delay(self.config)

        self.get_modality()
        self.continuous_label_dim = self.get_selected_continuous_label_dim()
        self.dataset_info = load_pickle(os.path.join(self.dataset_path, "dataset_info.pkl"))
        self.data_arranger = self.init_data_arranger()
        if self.calc_mean_std:
            self.calc_mean_std_fn()

    def run(self):
        import gc
        import torch
        import os  # 确保导入了 os
        from base.utils import ensure_dir, plot_training_curves  # 🌟 确保从 utils 导入了绘图函数

        criterion = CCCLoss()

        for fold in iter(self.folds_to_run):
            save_path = os.path.join(self.save_path,
                                     self.experiment_name + "_" + self.model_name + "_" + self.stamp + "_" + str(
                                         fold) + "_" + self.emotion)
            ensure_dir(save_path)

            checkpoint_filename = os.path.join(save_path, "checkpoint.pkl")

            model = self.init_model()
            dataloaders = self.init_dataloader(fold)

            trainer_kwards = {'device': self.device, 'emotion': self.emotion, 'model_name': self.model_name,
                              'model': model, 'save_path': save_path, 'fold': fold,
                              'min_epoch': self.config['min_epoch'], 'max_epoch': self.config['max_epoch'],
                              'early_stopping': self.config['early_stopping'], 'scheduler': self.scheduler,
                              'learning_rate': self.learning_rate, 'min_learning_rate': self.min_learning_rate,
                              'patience': self.patience, 'batch_size': self.batch_size,
                              'criterion': criterion, 'factor': self.factor, 'verbose': True, 'milestone': 0,
                              'metrics': self.config['metrics'],
                              'load_best_at_each_epoch': self.config['load_best_at_each_epoch'],
                              'save_plot': self.config['save_plot']}

            trainer = Trainer(**trainer_kwards)

            parameter_controller = ResnetParamControl(trainer, gradual_release=self.gradual_release,
                                                      release_count=self.release_count,
                                                      backbone_mode=self.backbone_mode)

            checkpoint_controller = Checkpointer(checkpoint_filename, trainer, parameter_controller, resume=self.resume)

            if self.resume:
                trainer, parameter_controller = checkpoint_controller.load_checkpoint()
            else:
                checkpoint_controller.init_csv_logger(self.args, self.config)

            # ---- 开始训练 ----
            if not trainer.fit_finished:
                # 1. 接收 fit 函数可能返回的字典
                fit_returns = trainer.fit(dataloaders, parameter_controller=parameter_controller,
                                          checkpoint_controller=checkpoint_controller)

                # 🌟 [绘图代码插入点] 🌟
                print(f"\n📊 正在生成第 {fold} 折的训练曲线图...")
                curve_save_dir = os.path.join(save_path, 'plots')

                # 2. 智能寻找训练记录字典 (适配不同版本的写法)
                record_dict = None
                if isinstance(fit_returns, tuple) and len(fit_returns) > 1 and isinstance(fit_returns[1], dict):
                    record_dict = fit_returns[1]  # 通常作为第二个参数返回
                elif hasattr(trainer, 'train_record_dict'):
                    record_dict = trainer.train_record_dict

                # 3. 安全绘图
                if record_dict:
                    from base.utils import plot_training_curves
                    plot_training_curves(record_dict, curve_save_dir, fold)
                else:
                    print("⚠️ 提示: 没找到对应的记录字典，已安全跳过绘图，不影响整体训练进度！")

            if not trainer.fold_finished and 'test' in dataloaders:
                test_kwargs = {'dataloader_dict': dataloaders, 'epoch': None, 'partition': 'test'}
                trainer.test(checkpoint_controller, predict_only=0, **test_kwargs)
                checkpoint_controller.save_checkpoint(trainer, parameter_controller, save_path)

            # ==== 🧹 核心修复点：清理显存 ====
            del model
            # 必须在画完图后再 del trainer
            del trainer
            del dataloaders
            del parameter_controller
            del checkpoint_controller
            gc.collect()
            torch.cuda.empty_cache()
            print(f"\n【🧹 显存清理完毕】第 {fold} 折跑完，已彻底清空显卡，准备进入下一折！\n")
    def init_model(self):
        self.init_randomness()
        if self.model_name == 'tcn':
            model = my_temporal(model_name=self.model_name, num_inputs=self.bandpower_dim,
                                cnn1d_channels=self.cnn1d_channels, cnn1d_kernel_size=self.cnn1d_kernel_size,
                                cnn1d_dropout_rate=self.cnn1d_dropout, embedding_dim=self.lstm_embedding_dim,
                                hidden_dim=self.lstm_hidden_dim, lstm_dropout_rate=self.lstm_dropout,
                                output_dim=1)
        elif self.model_name == 'satcn':
            model = SA_TCN(
                model_name=self.model_name,
                cnn1d_channels=self.cnn1d_channels, cnn1d_kernel_size=self.cnn1d_kernel_size,
                num_eeg_chan=self.num_eeg_chan, freq=self.num_f,
                cnn1d_dropout_rate=self.cnn1d_dropout,
                output_dim=1
            )
        elif self.model_name == 'masatcn':
            model = MSA_TCN(
                model_name=self.model_name,
                cnn1d_channels=self.cnn1d_channels, cnn1d_kernel_size=self.cnn1d_kernel_size,
                num_eeg_chan=self.num_eeg_chan, freq=self.num_f,
                cnn1d_dropout_rate=self.cnn1d_dropout,
                output_dim=1
            )

        return model

    def init_data_arranger(self):
        arranger = DataArranger(self.dataset_info, self.dataset_path, self.debug, self.task, self.case, self.seed)
        return arranger

    def init_dataset(self, data, continuous_label_dim, mode, fold):
        dataset = MyDatasetPreLoad(data, continuous_label_dim, self.modality, self.multiplier,
                          self.feature_dimension, self.window_length,
                          mode, mean_std=None, time_delay=self.time_delay)
        return dataset

    def get_modality(self):
        pass

    def get_config(self):
        from configs import config
        return config

    def get_selected_continuous_label_dim(self):
        dim = 0
        return dim
