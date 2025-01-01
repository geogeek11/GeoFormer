import hydra
from omegaconf import DictConfig, OmegaConf
import time
import os
from uuid import uuid4
import torch
from torch.utils.data import DataLoader
from src.dataloader import common
from src.pipelines.inference import load_model_from_chkpt
import datetime
import logging
from src.utils import data_utils
from src.utils.utils import AggregatedPredictionWriterCallback, exists, load_pickle
from src.pipelines.inference import compute_map
from hydra.utils import get_original_cwd
import wandb
from torch import distributed as dist

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import seed_everything


@hydra.main(config_path="./config/", config_name="inference.yaml")
def main(cfg: DictConfig) -> None:

    ########################################################
    ########### Configuration initialization ###############
    ########################################################
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)  

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "[%(asctime)s - %(name)s - %(levelname)s - %(message)s]"
    )
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

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

    collate_fn = data_utils.collate_fn_multipolygon

    ########################################################
    ######### Task to perform (training/test) ##############
    ########################################################

    if rank == 0:
        unq_str = str(uuid4())[:8]

        mod = [x for x in cfg.model.keys()][0]
        run_name = (
            next(iter(cfg.dataset))
            + "-"
            + unq_str
            + "-"
            + mod
            + "-"
            + cfg.meta.appendix
        )

        # Create a new directory with the name of the run
        outckpt_dir = f"./ckpts/{cfg.meta.chkpt_local_path_name}"
        if not os.path.exists(outckpt_dir):
            os.mkdir(outckpt_dir)

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

    logger.info(f"LOADING MODEL FROM CHECKPOINT {cfg.meta.chkpt_local_path_name}")
    if cfg.meta.task == "produce_inference_samples":
        plyformer, model_config = load_model_from_chkpt(
            out_ckpt_dir="./ckpts/",
            model_id=cfg.meta.chkpt_local_path_name,
            use_latest=cfg.meta.chkpt_use_latest,
            cfg=cfg if cfg.meta.use_untrained_config else None,
        )
        cfg.model = model_config.model
        plyformer.greedy = cfg.meta.inf_type

    logger.info(f"INFERENCE TYPE: {cfg.meta.inf_type}")

    ext_embedings = cfg.model.transformer_xformer.get("custom_embeddings_params").get(
        "type_of_embeddings"
    )

    if "global" in ext_embedings and cfg.model.transformer_xformer.get(
        "custom_embeddings"
    ):
        global_context = True
    else:
        global_context = False

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

    ## Defining writer callback
    ds_name = next(iter(cfg.dataset))
    inf_samples = (
        cfg.meta.max_num_inf_samples
        if cfg.meta.max_num_inf_samples
        else len(test_dataset)
    )
    predictions_save_path = f"{outckpt_dir}/{cfg.meta.chkpt_local_path_name}_{ds_name}_{inf_samples}_{cfg.meta.inf_type}_{cfg.meta.seed}_inference_samples.pkl"

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

    if cfg.meta.task == "produce_inference_samples":

        logger.info(
            f"\n Computing inference for {cfg.meta.max_num_inf_samples} batches \n"
        )

        t0 = time.time()

        plyformer.greedy = cfg.meta.inf_type == "greedy"
        ## Running the inference
        _ = trainer.predict(
            model=plyformer,
            dataloaders=test_dataset,
        )
        logger.info(f"Inference completed in {time.time()-t0} seconds")

    elif cfg.meta.task == "compute_metrics":
        logger.info("COMPUTING MAP SCORES")

        t0 = time.time()
        provided_save_path = cfg.meta.get("provided_save_path", False)
        if not provided_save_path:
            preds_array = load_pickle(predictions_save_path)
        else:
            logger.info(f"Loading provided predictions from {provided_save_path}")
            preds_array = load_pickle(cfg.meta.provided_save_path)

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
        metrics_computed["num_samples"] = len(
            _
        )  # cfg.meta.max_num_inf_samples if cfg.meta.max_num_inf_samples else len(test_dataset)
        metrics_computed["timestamp"] = datetime.datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        # Convert the DataFrame to a wandb Table
        table = wandb.Table(dataframe=metrics_computed)

        # Log the table
        wandb.log({"metrics": table})

        for i in range(cfg.meta.num_visual_inference_samples):
            pass  # Not implemented yet

        wandb.finish()


if __name__ == "__main__":
    main()
