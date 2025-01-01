# from src.utils.metrics import Metrics
import matplotlib.pyplot as plt

from x_transformers.autoregressive_wrapper import top_k

from pytorch_lightning import Callback


import wandb
import torch
import math


from src.pipelines.inference import plot_sequences


class InferencePredictionsLoggerXFormer(Callback):
    def __init__(
        self, val_batch, train_batch, greedy=True, external_embeddings=False, **kwargs
    ):
        super().__init__()
        self.val_batch = val_batch
        self.train_batch = train_batch

        self.greedy_inference = greedy
        self.external_embeddings = external_embeddings
        self.kwargs = kwargs

        if kwargs.get("stop_token"):
            self.stop_token = kwargs["stop_token"]
        else:
            self.stop_token = 1

        if kwargs.get("max_seq_len"):
            self.max_seq_len = kwargs["max_seq_len"]
        else:
            self.max_seq_len = 100

        if kwargs.get("multi_object_embeds", False):
            self.local_stop_token = 1
        else:
            self.local_stop_token = None

        self.num_obj_embeds = kwargs.get("num_obj_embeds", 50)

    def plot_samples_with_hits(self, d, pred_dist, grid_size=(4, 4)):

        prd_samples = pred_dist.probs
        gt_samples = d["vertices_flat"][1:]
        num_samples = prd_samples.shape[1]  # Batch dimension

        plt.figure(figsize=(15, 15))
        fig, axs = plt.subplots(grid_size[0], grid_size[1], figsize=(15, 15))
        axs = axs.flatten()  # Flatten the grid of axes to easily iterate over

        for i, ax in enumerate(axs):
            if i >= num_samples:  # If we have more plots than data, break
                break

            max_seq_len = len(gt_samples[d["vertices_flat_mask"][1:, i], i])
            hits = []
            hit = 0
            for v in range(max_seq_len):
                pred_value = prd_samples[v, i, :].detach().cpu().numpy()
                pred_vtx = pred_value.argmax()
                gt_value = gt_samples[v, i].detach().cpu().numpy()
                if int(pred_vtx == gt_value):
                    hit += 1
                    hits.append(pred_vtx)
                ax.plot(pred_value, "r-")
                ax.plot(pred_vtx, (pred_value.max() + 0.025), "rx")
                ax.plot(gt_value, (pred_value.max() + 0.025), "gx")

            ax.set_ylim([0, 0.25])
            ax.legend(
                [f"Num hits: \n {hit},{hits}", "pred token", "gt token"],
                loc="lower right",
            )

        plt.tight_layout()

    def plot_samples_grid(self, d, pred_samples):
        N = len(d["image"])  # number of samples
        cols = rows = math.isqrt(N)  # number of rows and columns

        plt.figure(figsize=(15, 15))
        fig, axes = plt.subplots(rows, cols, figsize=(10, 10))

        for i, ax in enumerate(axes.flat):
            # Plot image
            ax.imshow(d["image"][i].transpose(0, -1))
            ax.axis("off")

            # Get prediction and stop index
            samp = pred_samples[i, :]
            try:
                stop_idx = torch.where(samp == 1)[0][0]
            except:
                continue
            subsamp = samp[:stop_idx]

            gt_vertices = d["vtx_list"][i]
            try:
                vtxes = subsamp.reshape(len(subsamp) // 2, 2).cpu().numpy()
            except:
                vtxes = (
                    subsamp[: stop_idx - 1]
                    .reshape((stop_idx - 1) // 2, 2)
                    .cpu()
                    .numpy()
                )

            # Plot prediction and ground truth
            ax.plot(vtxes[:, 1], vtxes[:, 0], "rx-")

            if gt_vertices is list:
                for gt_vtx in gt_vertices:
                    ax.plot(gt_vtx[:, 1], gt_vtx[:, 0], "go-")
            else:
                ax.plot(gt_vertices[:, 1], gt_vertices[:, 0], "go-")

        plt.tight_layout()

    def plot_multi_object_samples(
        self, batch, preds, local_stop_token=1, num_spc_tokens=4
    ):
        N = len(batch["image"])
        cols = rows = math.ceil(math.sqrt(N))
        fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
        plt.subplots_adjust(hspace=0, wspace=0)

        for i, ax in enumerate(axes.flat):
            if i < N:
                try:
                    plot_sequences(
                        batch,
                        preds,
                        num_batch=i,
                        eos_token=local_stop_token,
                        num_spc_tokens=num_spc_tokens,
                        ax=ax,
                    )
                except Exception as e:
                    print(f"Exception when plotting sequences: {e}")
                    continue
            else:
                ax.axis("off")

        plt.tight_layout(pad=0)
        plt.show()

    def on_validation_epoch_end(self, trainer, pl_module):
        self._experiment = trainer.logger.experiment

        if pl_module.current_epoch > 0:
            with torch.no_grad():

                pl_module.eval()

                image_context = pl_module.produce_image_context(
                    self.val_batch["image"].to(pl_module.device)
                )

                gen_samples = pl_module.generate(
                    start_tokens=self.val_batch["vertices_flat"][[0], :]
                    .to(pl_module.device)
                    .transpose(0, 1),
                    seq_len=torch.tensor([self.max_seq_len]).to(pl_module.device),
                    eos_token=torch.tensor([self.stop_token]).to(pl_module.device),
                    local_eos_token=(
                        torch.tensor([self.local_stop_token]).to(pl_module.device)
                        if self.local_stop_token
                        else None
                    ),
                    temperature=1.0,
                    filter_logits_fn=top_k,
                    filter_thres=0.9,
                    min_p_pow=2.0,
                    min_p_ratio=0.02,
                    context=image_context,
                    greedy=self.greedy_inference,
                    external_embeddings=self.external_embeddings,
                    num_object_embeddings=self.kwargs.get("num_obj_embeds", 0),
                    multi_object=self.kwargs.get("multi_object_embeds"),
                )

                if self.kwargs.get("multi_object_embeds"):
                    tgt_sequence = (
                        self.val_batch["vertices_flat"].transpose(0, 1).cuda()
                    )
                    global_context = (
                        self.val_batch["vert_obj_embeds"].transpose(0, 1).cuda()
                    )
                    tgt_embeddings = pl_module.decoder.net.ext_embed(
                        tgt_sequence[:, :-1], global_context[:, :-1]
                    )
                else:
                    tgt_embeddings = None

                ls, logits = pl_module.decoder(
                    self.val_batch["vertices_flat"]
                    .to(pl_module.device)
                    .transpose(0, 1),
                    context=pl_module.produce_image_context(
                        self.val_batch["image"].to(pl_module.device)
                    ),
                    sum_embeds=tgt_embeddings,
                    return_logits=True,
                )

                alt_samples = logits.argmax(-1)

            if self.kwargs.get("multi_object_inference", True):
                self.plot_multi_object_samples(self.val_batch, alt_samples)
            else:
                self.plot_samples_grid(d=self.val_batch, pred_samples=alt_samples)

            trainer.logger.experiment.log(
                {"Inference diagnostics (non-AR)": wandb.Image(plt)}, commit=False
            )

            # Close the figure
            plt.close("all")

            if self.kwargs.get("multi_object_inference", True):
                self.plot_multi_object_samples(self.val_batch, gen_samples)
            else:
                self.plot_samples_grid(d=self.val_batch, pred_samples=gen_samples)

            trainer.logger.experiment.log(
                {"Inference (valid)": wandb.Image(plt)}, commit=False
            )

            # Close the figure
            plt.close("all")
