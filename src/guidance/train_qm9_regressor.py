# Rdkit import should be first, do not move it
from rdkit import Chem

import torch
import wandb
import hydra
import omegaconf
from omegaconf import DictConfig
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.warnings import PossibleUserWarning
import warnings

import src.utils as utils
import src.datasets.qm9_dataset as qm9_dataset

from src.metrics.molecular_metrics import SamplingMolecularMetrics
from src.metrics.molecular_metrics_discrete import  TrainMolecularMetricsDiscrete
from src.analysis.visualization import MolecularVisualization
from src.diffusion.extra_features import DummyExtraFeatures, ExtraFeatures
from src.diffusion.extra_features_molecular import ExtraMolecularFeatures
from src.guidance.qm9_regressor_discrete import Qm9RegressorDiscrete

warnings.filterwarnings("ignore", category=PossibleUserWarning)


def setup_wandb(cfg):
    config_dict = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    kwargs = {'name': cfg.general.name, 'project': 'graph_ddm_regressor', 'config': config_dict,
              'settings': wandb.Settings(_disable_stats=True),
              'reinit': True, 'mode': cfg.general.wandb}
    wandb.init(**kwargs)
    wandb.save('*.txt')
    return cfg


@hydra.main(version_base='1.1', config_path='../../configs', config_name='config')
def main(cfg: DictConfig):
    dataset_config = cfg["dataset"]
    print(cfg)
    
    print("get_num_threads", torch.get_num_threads())
    torch.set_num_threads(cfg.train.num_interops_workers)
    torch.set_num_interop_threads(cfg.train.num_interops_workers)
    print("get_num_threads", torch.get_num_threads())

    seed_everything(cfg.train.seed)

    assert dataset_config["name"] in ['qm9', 'zinc250k']
    assert cfg.model.type == 'discrete'

    if dataset_config["name"] == 'qm9':
        datamodule = qm9_dataset.QM9DataModule(cfg, regressor=True)
        dataset_infos = qm9_dataset.QM9infos(datamodule=datamodule, cfg=cfg)
        datamodule.prepare_data()
        train_smiles = None
    elif dataset_config["name"] == 'zinc250k':
        from src.datasets import zinc250k_dataset
        datamodule = zinc250k_dataset.ZINC250KDataModule(cfg)
        dataset_infos = zinc250k_dataset.ZINC250Kinfos(datamodule=datamodule, cfg=cfg)
        datamodule.prepare_data()
        train_smiles = None
    else:
        raise ValueError("Dataset not implemented")

    if cfg.model.extra_features is not None:
        extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
        domain_features = ExtraMolecularFeatures(dataset_infos=dataset_infos)
    else:
        extra_features = DummyExtraFeatures()
        domain_features = DummyExtraFeatures()

    dataset_infos.compute_input_output_dims(datamodule=datamodule, extra_features=extra_features,
                                            domain_features=domain_features)
    dataset_infos.output_dims = {'X': 0, 'E': 0, 'y': 2 if cfg.guidance.guidance_target == 'both' else 1}

    train_metrics = TrainMolecularMetricsDiscrete(dataset_infos)

    sampling_metrics = SamplingMolecularMetrics(dataset_infos, train_smiles)
    visualization_tools = MolecularVisualization(cfg.general.remove_h, dataset_infos=dataset_infos)

    model_kwargs = {'dataset_infos': dataset_infos, 'train_metrics': train_metrics,
                    'sampling_metrics': sampling_metrics, 'visualization_tools': visualization_tools,
                    'extra_features': extra_features, 'domain_features': domain_features}

    utils.create_folders(cfg)
    cfg = setup_wandb(cfg)

    model = Qm9RegressorDiscrete(cfg=cfg, **model_kwargs)

    callbacks = []
    if cfg.train.save_model:
        checkpoint_callback = ModelCheckpoint(dirpath=f"checkpoints/{cfg.general.name}",
                                              filename='{epoch}',
                                              monitor='val/epoch_mae',
                                              save_last=True,
                                              save_top_k=3,    # was 5
                                              mode='min',
                                              every_n_epochs=cfg.train.ckpt_every_n_train_steps,
                                              save_on_train_epoch_end = True)
        callbacks.append(checkpoint_callback)

    name = cfg.general.name
    if name == 'debug':
        print("[WARNING]: Run is called 'debug' -- it will run with fast_dev_run. ")
    
    if(isinstance(cfg.general.gpus, int)):
        gpus_ok = cfg.general.gpus >= 0
        n_gpus = cfg.general.gpus
    elif(isinstance(cfg.general.gpus, omegaconf.ListConfig)):
        gpus_ok = True
        n_gpus = 0
        for gpu in cfg.general.gpus:
            if(gpu < 0): gpus_ok = False
            n_gpus = n_gpus + 1
    else:
        gpus_ok = False
        n_gpus = 0
    
    print("cfg.general.gpus", cfg.general.gpus)
    print("torch.cuda.is_available()", torch.cuda.is_available())

    use_gpu = torch.cuda.is_available() and gpus_ok
    print("gpus_ok", gpus_ok)
    print("use_gpu", use_gpu)
    
    trainer = Trainer(gradient_clip_val=cfg.train.clip_grad,
                      accelerator='gpu' if use_gpu else 'cpu',
                      devices=cfg.general.gpus if use_gpu else None,
                      
                      limit_train_batches=20 if name == 'test' else None,
                      limit_val_batches=20 if name == 'test' else None,
                      limit_test_batches=20 if name == 'test' else None,
                      val_check_interval=cfg.general.val_check_interval,
                      max_epochs=cfg.train.n_epochs,
                      check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,
                      fast_dev_run=cfg.general.name == 'debug',

                      enable_progress_bar = cfg.train.progress_bar,
                      callbacks=callbacks,
                      logger=[])

    trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.general.resume)


if __name__ == '__main__':
    main()
