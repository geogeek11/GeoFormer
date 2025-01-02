import hydra
from omegaconf import DictConfig, OmegaConf
import time
import os
from uuid import uuid4
import torch

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

from torch.utils.data import DataLoader

from src.dataloader import common
from src.utils import data_utils
from src.utils.utils import exists, load_pickle, AggregatedPredictionWriterCallback
from src.models.pl_modules import InferencePredictionsLoggerXFormer
from src.pipelines import initializer
from src.models.xtransformer import XTransformerTrainer
from src.pipelines.inference import compute_map,load_model_from_chkpt

import logging
from hydra.utils import get_original_cwd

import wandb
import torch
from torch import distributed as dist

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import datetime
from pytorch_lightning import seed_everything

import torch


@hydra.main(config_path="./config/", config_name="geoformer_meta.yaml")
def main(cfg: DictConfig) -> None:
    ########################################################
    ############## Logging initialization ##################
    ########################################################

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)  # Set the logging level to INFO
    # Create a console handler and set its level to INFO
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create a formatter and add it to the handler
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(console_handler)

    ########################################################
    ########### Configuration initialization ###############
    ########################################################

    print(OmegaConf.to_yaml(cfg))
    wconf = OmegaConf.to_container(cfg, resolve=[True | False])

    os.environ["WANDB_START_METHOD"] = "thread"

    wandb_logger = None

    # Check if distributed is available and get the rank
    is_distributed = (
        torch.distributed.is_available() and torch.distributed.is_initialized()
    )
    rank = dist.get_rank() if is_distributed else 0

    ########################################################
    ############## Dataset initialization ##################
    ########################################################

    seed_everything(cfg.meta.seed)

    root_path = get_original_cwd()
    os.environ["WANDB_DIR"] = "./wandb/"

    ds, valid_ds, _ = common.dataset_loader(cfg, module_root_path=root_path)

    collate_fn = data_utils.collate_fn_multipolygon

    if cfg.meta.subset_training:
        ds = common.get_subset_dataset(ds, num_samples=cfg.meta.num_examples_in_subset)
        valid_ds = common.get_subset_dataset(valid_ds, num_samples=1000)

    # Random shuffle the dataset
    ds_key = next(iter(cfg.dataset))
    random_shuffle_dataset = cfg.dataset.get(ds_key).get("random_shuffle", False)
    random_cycle_dataset = cfg.dataset.get(ds_key).get("cycle_start_token", False)

    if not cfg.meta.task == "inference":
        logger.info(
            f"SHUFFING DATASET: {random_shuffle_dataset}, CYCLE START TOKEN: {random_cycle_dataset}"
        )

        ds = DataLoader(
            ds,
            batch_size=cfg.meta.batch_size,
            collate_fn=lambda x: (
                collate_fn(
                    x,
                    random_shuffle=random_shuffle_dataset,
                    cycle_start_token=random_cycle_dataset,
                )
            ),
            shuffle=True,
            drop_last=False,
            pin_memory=True,
            num_workers=cfg.meta.num_workers,
        )

        valid_ds = DataLoader(
            valid_ds,
            batch_size=cfg.meta.valid_batch_size,
            collate_fn=collate_fn,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            num_workers=int(cfg.meta.num_workers // 2),
        )

        train_batch = next(iter(ds))
        val_batch = next(iter(valid_ds))

    ########################################################
    ############## Model initialization ####################
    ########################################################

    encoder, decoder = initializer.get_model(cfg)

    ext_embedings = cfg.model.transformer_xformer.get("custom_embeddings_params").get(
        "type_of_embeddings"
    )

    if "global" in ext_embedings and cfg.model.transformer_xformer.get(
        "custom_embeddings", False
    ):
        print(f"Using global context embeddings: {ext_embedings} \n")
    if "global" in ext_embedings and cfg.model.transformer_xformer.get(
        "custom_embeddings"
    ):
        global_context = True
    else:
        global_context = False

    plyformer = XTransformerTrainer(
        encoder=encoder,
        decoder=decoder,
        lr=cfg.meta.learning_rate,
        warmup_steps=cfg.meta.warmup_steps,
        encoder_backbone=cfg.model.transformer_xformer.backbone,
        global_context=global_context,
        normalize_images=cfg.model.transformer_xformer.get("normalize_images", False),
    )

    ########################################################
    ######### Task to perform (training/test) ##############
    ########################################################

    unq_str = str(uuid4())[:8]

    mod = [x for x in cfg.model.keys()][0]
    dataset_name = next(iter(cfg.dataset))
    run_name = (
        unq_str
        + "-"
        + str(dataset_name)
        + "-"
        + mod
        + "-"
        + cfg.model.get(mod).backbone
        + "-"
        + str(cfg.model.get(mod).swin_params.output_hidden_dims)
        + "-"
        + "pyramid_"
        + str(cfg.model.get(mod).swin_params.pyramidal_ft_maps)
        + "-"
        + "spatial_"
        + str(cfg.model.get(mod).decoder.custom_spatial_abs_embedding)
        + "-"
        + "num_layers_"
        + str(cfg.model.get(mod).dec_attnlayers.depth)
        + "-"
        + "num_heads_"
        + str(cfg.model.get(mod).dec_attnlayers.heads)
        + "-"
        + "alibi_heads_"
        + str(cfg.model.get(mod).dec_attnlayers.alibi_num_heads)
        + "-"
        + "rope_"
        + str(cfg.model.get(mod).dec_attnlayers.rotary_pos_emb)
        + "-"
        + "mask_"
        + str(cfg.model.get(mod).misc_params.mask_prob)
        + "-"
        + str(cfg.meta.appendix)
    )

    outckpt_dir = f"./ckpts/{run_name}"
    os.mkdir(outckpt_dir)

    # save config for future reference
    with open(f"{outckpt_dir}/{run_name}.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    if "train" in cfg.meta.task:
        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            patience=cfg.meta.patience,
        )

        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            save_top_k=5,
            save_last=True,
            save_weights_only=False,
            verbose=True,
            mode="min",
            every_n_epochs=1,
            dirpath=outckpt_dir,
            filename="{epoch}-{val_loss:.4f}",
        )

        # Only initialize WandbLogger on rank 0
        if rank == 0:
            wandb_logger = WandbLogger(
                log_model=True,
                project="BMVC-2024",
                name=run_name,
                group=cfg.meta.group,
                job_type=cfg.meta.job_type,
                config=wconf,
                reinit=True,
            )
            wandb_logger.watch(plyformer)
        else:
            wandb_logger = None

        grad_clip = cfg.meta.grad_clipping_val

        trainer = Trainer(
            max_epochs=cfg.meta.num_epochs,
            devices=cfg.meta.num_gpus,
            check_val_every_n_epoch=cfg.meta.check_val_n_epochs,
            strategy=cfg.meta.distributed_backend,
            enable_checkpointing=checkpoint_callback,
            callbacks=[
                early_stop_callback,
                InferencePredictionsLoggerXFormer(
                    val_batch=val_batch,
                    train_batch=train_batch,
                    greedy=cfg.meta.inf_type,
                    external_embeddings=cfg.model.transformer_xformer.get(
                        "custom_embeddings"
                    ),
                    multi_object_visuals=True,
                    multi_object_embeddings=global_context,
                    stop_token=3,
                    max_seq_len=cfg.model.transformer_xformer.decoder.get(
                        "max_seq_len", 100
                    ),
                    num_obj_embeds=cfg.model.transformer_xformer.get(
                        "custom_embeddings_params"
                    ).get("num_obj_embeds", 0),
                ),
                checkpoint_callback,
            ],
            logger=wandb_logger,
            gradient_clip_val=grad_clip,
            gradient_clip_algorithm="norm",
            precision=cfg.meta.precision,
            max_steps=cfg.meta.get("max_steps", -1),
        )

        
        if cfg.meta.restore_from_checkpoint:
            plyformer, model_config = load_model_from_chkpt(
                out_ckpt_dir="./ckpts/",
                model_id=cfg.meta.chkpt_local_path_name,
                use_latest=cfg.meta.chkpt_use_latest,
                cfg=cfg if cfg.meta.use_untrained_config else None,
            )
            cfg.model = model_config.model
            plyformer.greedy = cfg.meta.inf_type

            trainer.fit(plyformer, ds, valid_ds)
        else:
            trainer.fit(plyformer, ds, valid_ds)

        run_name = run_name

        
        trainer.save_checkpoint(f"{outckpt_dir}/{run_name}.ckpt")

        if "test" not in cfg.meta.task:
            wandb.finish()

        #####################################################
        ## Commands from "Improving Hydra+DDP support #11617"
        if dist.is_initialized():
            dist.destroy_process_group()

        envs = (
            "LOCAL_RANK",
            "NODE_RANK",
            "WORLD_SIZE",
            "MASTER_ADDR",
            "MASTER_PORT",
            "PL_GLOBAL_SEED",
            "PL_SEED_WORKERS",
        )

        for name in envs:
            os.environ.pop(name, None)

    if "test" in cfg.meta.task:
        logger.info(
            f"PROCEEDING WITH INFERENCE ON TEST SET {cfg.meta.chkpt_local_path_name}"
        )

        if not exists(wandb_logger):
            # Only initialize WandbLogger on rank 0
            if rank == 0:
                wandb.init(
                    project="BMVC-2024",
                    name=run_name,
                    group="INFERENCE-METRICS",
                    job_type=cfg.meta.task,
                    config=wconf,
                    reinit=True,
                )

                wandb_logger = WandbLogger(
                    log_model=False,
                    project="BMVC-2024",
                    name=run_name,
                    group="INFERENCE-METRICS",
                    job_type=cfg.meta.task,
                    config=wconf,
                    reinit=True,
                )

        logger.info(f"INFERENCE TYPE: {cfg.meta.inf_type}")

        _, _, test_dataset = common.dataset_loader(cfg, module_root_path=root_path)

        test_dataset = DataLoader(
            test_dataset,
            batch_size=cfg.meta.valid_batch_size,
            collate_fn=collate_fn,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            num_workers=int(cfg.meta.num_workers),
        )

        inf_samples = (
            cfg.meta.max_num_inf_samples
            if cfg.meta.max_num_inf_samples
            else len(test_dataset)
        )
        predictions_save_path = f"{outckpt_dir}/{run_name}_{inf_samples}_{cfg.meta.inf_type}_{cfg.meta.seed}_inference_samples.pkl"

        prediction_writer = AggregatedPredictionWriterCallback(
            save_path=predictions_save_path
        )

        # Initialize the trainer
        trainer = Trainer(
            devices=cfg.meta.num_gpus,
            logger=wandb_logger,
            precision=cfg.meta.precision,
            limit_predict_batches=cfg.meta.max_num_inf_samples,
            callbacks=[prediction_writer],
        )

        single_batch = next(iter(test_dataset))

        if not cfg.meta.max_num_inf_samples:
            cfg.meta.max_num_inf_samples = len(test_dataset)

        logger.info(
            f"\n Computing inference for {cfg.meta.max_num_inf_samples} batches \n"
        )

        t0 = time.time()

        ## Running the inference
        _ = trainer.predict(
            model=plyformer,
            dataloaders=test_dataset,
        )
        logger.info(f"Inference completed in {time.time()-t0} seconds")
        logger.info(f"Predictions saved at: {predictions_save_path}")

        if "inference" in cfg.meta.task:
            logger.info("COMPUTING MAP SCORES")

            preds_array = load_pickle(predictions_save_path)

            metrics_computed = compute_map(
                preds_array,
                image_size=single_batch["image"].shape[-1],
                compute_by_object=True,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )

            metrics_computed["model"] = run_name  # cfg.meta.chkpt_local_path_name
            metrics_computed["dataset"] = next(iter(cfg.dataset))
            metrics_computed["inference_type"] = cfg.meta.inf_type
            metrics_computed["time_taken"] = time.time() - t0
            metrics_computed["time_stamp"] = datetime.datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            metrics_computed["num_samples"] = len(
                _
            )  

            table = wandb.Table(dataframe=metrics_computed)
            
            wandb.log({"metrics": table})
            
            metrics_computed.to_csv(
                f"{outckpt_dir}/{run_name}_{inf_samples}_{cfg.meta.inf_type}_{cfg.meta.seed}_metrics.csv"
            )

            for i in range(cfg.meta.get("num_visual_inference_samples", 0)):
                pass  # Not implemented

        wandb.finish()


if __name__ == "__main__":
    main()
