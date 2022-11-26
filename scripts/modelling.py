import torch
import pytorch_lightning as pl
from transformers import AutoModel
from sentence_transformers.models import Pooling
from pytorch_metric_learning import miners, losses


class CoherenceAwareSentenceEmbedder(pl.LightningModule):
    def __init__(
        self,
        mpath: str,
        num_classes: int,
        triplet_margin: float = 0.5,
        surrogate_imp: float = 0.5,
        embedder_lr: float = 1e-5,
        surrogate_lr: float = 1e-3,
        weight_decay: float = 1e-4
    ):
        """
        Args:
            mpath (str): Path to the `transformer` artifact files.
            num_classes (int): Number of classes in the data.
            triplet_margin (float): The margin to be used in triplet loss.
            surrogate_imp (float): How much importance range(0-1) to assign
            to the surrogate loss, 0 would mean no surroagte loss and 1 would
            mean the semantic loss is ignored.
            embedder_lr (float, optional): Embedder's learning rate.
            Defaults to 1e-5.
            surrogate_lr (float, optional): Surrogate model's learning rate.
            Defaults to 1e-3.
            weight_decay (float, optional): Common weight decay co-efficient.
            Defaults to 1e-4.
        """
        super().__init__()
        self.surrogate_imp = surrogate_imp
        self.embedder_lr = embedder_lr
        self.surrogate_lr = surrogate_lr
        self.weight_decay = weight_decay

        # Base model        
        self.embedder = AutoModel.from_pretrained(mpath)
        # Pooling layer to get the sentence embedding
        self.pooling = Pooling(
            self.embedder.config.hidden_size,
            pooling_mode='mean'
        )

        # Next sentece label predictor
        self.surrogate_head = torch.nn.Linear(
            self.embedder.config.hidden_size,
            2
        )

        # To mine the triplets
        self.miner = miners.BatchHardMiner()
        # Loss function for next sentence class prediction
        self.surr_loss = torch.nn.BCEWithLogitsLoss()
        # Loss function: Fixed for now
        self.sem_loss = losses.TripletMarginLoss(margin=triplet_margin)

        # Save the init arguments
        self.save_hyperparameters()

    def forward(self, **kwargs):
        # Push all inputs to the device in use
        kwargs = {k: v.to(self.device) for k, v in kwargs.items()}
        token_embeddings = self.embedder(**kwargs)[0]

        sentence_embedding = self.pooling({
            'token_embeddings': token_embeddings,
            'attention_mask': kwargs['attention_mask']
        })['sentence_embedding']

        return sentence_embedding   

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

        # Get the surrogate task preds
        next_sent_labels_pred = self.surrogate_head(sent_embeddings)

        # Surrogate loss
        surr_loss =  self.surr_loss(
            next_sent_labels_pred,
            shifts
        )

        # Get the triplets
        triplet_indices = self.miner(
            sent_embeddings,
            sent_labels
        )

        # Semantic contrast loss
        sem_loss = self.sem_loss(sent_embeddings, indices_tuple=triplet_indices)

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

    def configure_optimizers(self):
        param_dicts = [
              {"params": self.surrogate_head.parameters()},
              {
                  "params": self.embedder.parameters(),
                  "lr": self.embedder_lr,
              },
        ]

        return torch.optim.AdamW(
            param_dicts,
            lr=self.surrogate_lr,
            weight_decay=self.weight_decay
        )
