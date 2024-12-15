import torch
from tqdm.auto import tqdm
from pathlib import Path
import wandb
from src.metrics.tracker import MetricTracker
from src.metrics.utils import calc_cer, calc_wer
from src.trainer.base_trainer import BaseTrainer
from src.logger.utils import plot_spectrogram
import pandas as pd

class Inferencer(BaseTrainer):
    """
    Inferencer (Like Trainer but for Inference) class

    The class is used to process data without
    the need of optimizers, writers, etc.
    Required to evaluate the model on the dataset, save predictions, etc.
    """

    def __init__(
            self,
            model,
            config,
            device,
            dataloaders,
            text_encoder,
            save_dir,
            writer,
            logger,
            metrics=None,
            batch_transforms=None,
            skip_model_load=False,
    ):
        """
        Initialize the Inferencer.

        Args:
            model (nn.Module): PyTorch model.
            config (DictConfig): run config containing inferencer config.
            device (str): device for tensors and model.
            dataloaders (dict[DataLoader]): dataloaders for different
                sets of data.
            text_encoder (CTCTextEncoder): text encoder.
            save_dir (str): path to save model predictions and other
                information.
            metrics (dict): dict with the definition of metrics for
                inference (metrics[inference]). Each metric is an instance
                of src.metrics.BaseMetric.
            batch_transforms (dict[nn.Module] | None): transforms that
                should be applied on the whole batch. Depend on the
                tensor name.
            skip_model_load (bool): if False, require the user to set
                pre-trained checkpoint path. Set this argument to True if
                the model desirable weights are defined outside of the
                Inferencer Class.
        """
        assert (
                skip_model_load or config.inferencer.get("from_pretrained") is not None
        ), "Provide checkpoint or set skip_model_load=True"

        self.config = config
        self.cfg_trainer = self.config.inferencer

        self.device = device
        self.logger = logger
        self.writer = writer
        self.model = model
        self.batch_transforms = batch_transforms

        self.text_encoder = text_encoder

        # define dataloaders
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items()}

        # path definition

        self.save_dir = save_dir

        # define metrics
        self.metrics = metrics
        if self.metrics is not None:
            self.evaluation_metrics = MetricTracker(
                *[m.name for m in self.metrics["inference"]],
                writer=None,
            )
        else:
            self.evaluation_metrics = None

        if not skip_model_load:
            # init model
            self._from_pretrained(config.inferencer.get("from_pretrained"))

    def run_inference(self):
        """
        Run inference on each partition.

        Returns:
            part_logs (dict): part_logs[part_name] contains logs
                for the part_name partition.
        """
        part_logs = {}
        for part, dataloader in self.evaluation_dataloaders.items():
            logs = self._inference_part(part, dataloader)
            part_logs[part] = logs
        return part_logs

    def _inference_part(self, part, dataloader):
        """
        Run inference on a given partition and save predictions

        Args:
            part (str): name of the partition.
            dataloader (DataLoader): dataloader for the given partition.
        Returns:
            logs (dict): metrics, calculated on the partition.
        """

        self.is_train = False
        self.model.eval()

        self.evaluation_metrics.reset()

        # create Save dir
        if self.save_dir is not None:
            (self.save_dir / part).mkdir(exist_ok=True, parents=True)

        with torch.no_grad():
            for batch_idx, batch in tqdm(
                    enumerate(dataloader),
                    desc=part,
                    total=len(dataloader),
            ):
                if "test" not in part:
                    raise Exception(f"Evaluating part without 'test' prefics. part name is {part}")

                batch = self.process_batch(
                    batch_idx=batch_idx,
                    batch=batch,
                    part=part,
                    metrics=self.evaluation_metrics,
                )

        return self.evaluation_metrics.result()

    def process_batch(self, batch_idx, batch, metrics, part):
        """
        Run batch through the model, compute metrics, and
        save predictions to disk.

        Save directory is defined by save_dir in the inference
        config and current partition.

        Args:
            batch_idx (int): the index of the current batch.
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type
                of the partition (train or inference).
            part (str): name of the partition. Used to define proper saving
                directory.
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform)
                and model outputs.
        """

        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        outputs = self.model(**batch)
        batch.update(outputs)

        if metrics is not None:
            for met in self.metrics["inference"]:
                metrics.update(met.name, met(**batch))

        # Some saving logic. This is an example
        # Use if you need to save predictions on disk
        self._log_batch(batch_idx, batch, mode="inference")
        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        # method to log data from you batch
        # such as audio, text or images, for example

        # logging scheme might be different for different partitions
        if mode == "train":  # the method is called only every self.log_step steps
            self.log_spectrogram(**batch)
        else:
            # Log Stuff

            self.log_spectrogram(**batch)
            self.log_predictions(**batch)
            self.log_audio(**batch)

    def log_spectrogram(self, spectrogram, spectrogram_raw, **batch):
        spectrogram_for_plot = spectrogram[0].detach().cpu()
        spectrogram_for_plot_raw = spectrogram_raw[0].detach().cpu()
        image = plot_spectrogram(spectrogram_for_plot)
        image_raw = plot_spectrogram(spectrogram_for_plot_raw)
        self.writer.add_image("spectrogram", image)
        self.writer.add_image("spectrogram_raw", image_raw)

    def log_audio(self, audio, raw_audio, audio_path, **batch):

        rows_audio = {}

        rows_audio[Path(audio_path).name] = {
            "raw_audio": wandb.Audio(raw_audio, sample_rate=16000),
            "audio": wandb.Audio(audio, sample_rate=16000)
        }

        self.writer.add_table(
            "audio", pd.DataFrame.from_dict(rows_audio, orient="index")
        )


    def log_predictions(
        self, text, log_probs, log_probs_length, audio_path, examples_to_log=100, **batch
    ):

        argmax_inds = log_probs.cpu().argmax(-1).numpy()
        argmax_inds = [
            inds[: int(ind_len)]
            for inds, ind_len in zip(argmax_inds, log_probs_length.cpu().numpy())
        ]
        argmax_texts_raw = [self.text_encoder.decode(inds) for inds in argmax_inds]
        argmax_texts = [self.text_encoder.ctc_decode(inds) for inds in argmax_inds]
        predictions_bs = [
            self.text_encoder.ctc_beam_search(log_prob[:len])
            for log_prob, len in zip(log_probs, log_probs_length)
        ]
        tuples = list(zip(predictions_bs, argmax_texts, text, argmax_texts_raw, audio_path))

        rows = {}
        for beam_search_predictions, argmax_predictions, target, raw_pred, audio_path in tuples[:examples_to_log]:
            target = self.text_encoder.normalize_text(target)
            wer_beam_search = calc_wer(target, beam_search_predictions) * 100
            cer_beam_search = calc_cer(target, beam_search_predictions) * 100
            wer_argmax = calc_wer(target, argmax_predictions) * 100
            cer_argmax = calc_cer(target, argmax_predictions) * 100

            rows[Path(audio_path).name] = {
                "target": target,
                "raw prediction": raw_pred,
                "argmax_predictions": argmax_predictions,
                "beam_search_predictions": beam_search_predictions,
                "wer_argmax": wer_argmax,
                "cer_argmax": cer_argmax,
                "wer_beam_search": wer_beam_search,
                "cer_beam_search": cer_beam_search,
                "audio": wandb.Audio(audio_path)
            }

        self.writer.add_table(
            "predictions", pd.DataFrame.from_dict(rows, orient="index")
        )