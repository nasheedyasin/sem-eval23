import torch
import pytorch_lightning as pl

from transformers import AutoModel
from torch.nn import functional as F
from sentence_transformers.models import Pooling
from pytorch_metric_learning import miners, losses, distances


class SemanticInfuser(torch.nn.Module):
    def __init__(self,
        input_dim: int,
        embedding_dim: int,
        semantic_filters: int
    ):
        """
        Args:
            input_dim (int): Dimensionality of the input representation.
            embedding_dim (int): Dimensionality of the semantic (output)
            representation.
            semantic_filters (int, optional): Number of semantic features to
            extract.
        """
        super().__init__()

        # Semantic Embedder
        # calc the dimensionality of the conv kernel
        stride = 1
        kernel_dim = input_dim - stride * (embedding_dim - 1)

        assert kernel_dim > 0, "`embedding_dim` too big."

        self.infuser = torch.nn.Conv1d(
            in_channels=1,
            out_channels=semantic_filters,
            kernel_size=kernel_dim,
            stride=stride
        )

        # A 1-D conv layer to format the output. Function similar to 1x1 2d conv in
        # InceptionNet.
        self.presenter = torch.nn.Conv1d(
            in_channels=semantic_filters,
            out_channels=1,
            kernel_size=1,
            stride=1
        )

    def forward(self, plain_embedding):
        # Reshape to get add the channel dim
        plain_embedding = plain_embedding.unsqueeze(1)
        # Get the semantic embedding
        sem_embedding = self.infuser(plain_embedding)
        sem_embedding = self.presenter(sem_embedding)

        # Remove the channel dim
        return sem_embedding.squeeze(1)


class CoherenceAwareSentenceEmbedder(pl.LightningModule):
    def __init__(
        self,
        mpath: str,
        num_classes: int,
        embedding_dim: int = 128,
        semantic_filters: int = 4,
        triplet_margin: float = 1.0,
        surrogate_imp: float = 0.5,
        sem_infuser_lr: float = 2e-3,
        surrogate_lr: float = 1e-4,
        weight_decay: float = 1e-4
    ):
        """
        Args:
            mpath (str): Path to the `transformer` artifact files.
            num_classes (int): Number of classes in the data.
            embedding_dim (int): Dimensionality of the semantic representation.
            Defaults to 128.
            semantic_filters (int, optional): Number of semantic features to
            extract. Defaults to 4.
            triplet_margin (float): The margin to be used in triplet loss.
            surrogate_imp (float): How much importance range(0-1) to assign
            to the surrogate loss, 0 would mean no surroagte loss and 1 would
            mean the semantic loss is ignored.
            sem_infuser_lr (float, optional): Sem infuser's learning rate.
            Defaults to 2e-3.
            surrogate_lr (float, optional): Surrogate model's learning rate.
            Defaults to 1e-3.
            weight_decay (float, optional): Common weight decay co-efficient.
            Defaults to 1e-4.
        """
        super().__init__()
        self.surrogate_imp = surrogate_imp
        self.sem_infuser_lr = sem_infuser_lr
        self.surrogate_lr = surrogate_lr
        self.weight_decay = weight_decay

        # Base model        
        self.backbone = AutoModel.from_pretrained(mpath)
        # Pooling layer to get the sentence embedding
        self.pooling = Pooling(
            self.backbone.config.hidden_size,
            pooling_mode='mean'
        )
        # Freeze the backbone model
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Ensure that the semantic_dims are smaller than the backbone's hidden dim
        assert embedding_dim <= self.backbone.config.hidden_size,\
            "Ensure that `embedding_dim`is smaller than the transformer's "\
            "`hidden_dim`"

        # Semantic Infuser
        self.sem_infuser = SemanticInfuser(
            self.backbone.config.hidden_size,
            embedding_dim,
            semantic_filters
        )


        # Next sentece label predictor
        self.surrogate_head = torch.nn.Linear(
            embedding_dim,
            2
        )

        # To mine the triplets
        self.miner = miners.BatchEasyHardMiner(
            pos_strategy="hard",
            neg_strategy="semihard",
            distance=distances.LpDistance(normalize_embeddings=False)
        )
        # Loss function for next sentence class prediction
        self.surr_loss = torch.nn.BCEWithLogitsLoss()
        # Loss function: Fixed for now
        self.sem_loss = losses.TripletMarginLoss(
            margin=triplet_margin,
            distance=distances.LpDistance(normalize_embeddings=False)
        )

        # This is set to help with mining strategy callbacks
        self.triplet_margin = triplet_margin

        # Save the init arguments
        self.save_hyperparameters()

    def forward(self, **kwargs):
        # Push all inputs to the device in use
        kwargs = {k: v.to(self.device) for k, v in kwargs.items()}
        token_embeddings = self.backbone(**kwargs)[0]

        # Vanilla embedding from the backbone
        sentence_embedding = self.pooling({
            'token_embeddings': token_embeddings,
            'attention_mask': kwargs['attention_mask']
        })['sentence_embedding']

        # Infuse semantics
        return self.sem_infuser(sentence_embedding)

    def common_step(self, batch, batch_idx):
        """
        Args:
            batch (_type_): _description_
            batch_idx (_type_): _description_
            surr_loss_only (bool, optional): Indicates whether optimizing
            the surrogate head's parameters. Defaults to False.

        Returns:
            _type_: _description_
        """
        sent_tokens, (sent_labels, shifts) = batch
        sent_embeddings = self(**sent_tokens)

        # Get the surrogate task preds and loss
        surr_loss =  self.surr_loss(
            self.surrogate_head(sent_embeddings),
            shifts
        )

        # Get the triplets
        triplet_indices = self.miner(
            sent_embeddings,
            sent_labels
        )

        # Semantic contrast loss
        sem_loss = self.sem_loss(
            sent_embeddings,
            sent_labels,
            indices_tuple=triplet_indices
        )

        return {
            'loss': self.surrogate_imp * surr_loss + \
                (1-self.surrogate_imp) * sem_loss,
            'sem_loss': sem_loss,
            'surr_loss': surr_loss
        }

    def training_step(self, batch, batch_idx):
        loss_dict = self.common_step(batch, batch_idx)
        # logs metrics for each training_step,
        # and the average across the epoch
        for k,v in loss_dict.items():
            self.log("train_" + k, v.item(), prog_bar=True)

        return loss_dict

    def validation_step(self, batch, batch_idx):
        loss_dict = self.common_step(batch, batch_idx)
        # logs metrics for each training_step,
        # and the average across the epoch
        for k,v in loss_dict.items():
            self.log("val_" + k, v.item(), prog_bar=True)

        return loss_dict

    def predict_step(self, batch, batch_idx):
        sent_tokens, _ = batch

        with torch.no_grad():
            sent_embeddings = self(**sent_tokens)

        return sent_embeddings

    def configure_optimizers(self):
        param_dicts = [
              {"params": self.surrogate_head.parameters()},
              {
                  "params": self.sem_infuser.parameters(),
                  "lr": self.sem_infuser_lr,
              },
        ]

        optimizer = torch.optim.AdamW(
            param_dicts,
            lr=self.surrogate_lr,
            weight_decay=self.weight_decay
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    patience=4),
                "monitor": "val_sem_loss"
            },
        }


class AlterMiningStrategy(pl.Callback):
    def __init__(
        self,
        monitor: str,
        threshold_coeff: float = 0.6,
        miner: miners.BaseMiner = None,
        keep_active_during_sanity = False
    ):
        super().__init__()

        # To prevent from activating after sanity check
        self.active = keep_active_during_sanity

        self.monitor = monitor
        self.threshold_coeff = threshold_coeff

        if miner is None:
            self.miner = miners.BatchHardMiner(
                distance=distances.LpDistance(normalize_embeddings=False)
            )

    def on_sanity_check_end(self, trainer, pl_module):
        self.active = True

    def on_validation_end(self, trainer, pl_module):
        # Change mining strat when loss is <= half the margin
        if self.active and trainer.callback_metrics[self.monitor] <=\
            (self.threshold_coeff * pl_module.triplet_margin):
            pl_module.miner = self.miner
