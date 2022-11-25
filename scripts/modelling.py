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
        embedder_lr: float = 1e-5,
        surrogate_lr: float = 1e-3,
        weight_decay: float = 1e-4
    ):
        """
        Args:
            mpath (str): Path to the `transformer` artifact files.
            num_classes (int): Number of classes in the data.
            triplet_margin (float): The margin to be used in triplet loss.
            embedder_lr (float, optional): Embedder's learning rate.
            Defaults to 1e-5.
            surrogate_lr (float, optional): Surrogate model's learning rate.
            Defaults to 1e-3.
            weight_decay (float, optional): Common weight decay co-efficient.
            Defaults to 1e-4.
        """
        super().__init__()
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
            num_classes
        )

        # To mine the triplets
        self.miner = miners.BatchHardMiner()
        # Loss function for next sentence class prediction
        self.surr_loss = torch.nn.CrossEntropyLoss()
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

    def common_step(self, batch, batch_idx, surr_loss_only=False):
        """
        Args:
            batch (_type_): _description_
            batch_idx (_type_): _description_
            surr_loss_only (bool, optional): Indicates whether optimizing
            the surrogate head's parameters. Defaults to False.

        Returns:
            _type_: _description_
        """
        sent_tokens, (sent_labels, next_sent_labels) = batch
        sent_embeddings = self(**sent_tokens)

        # Get the surrogate task preds
        next_sent_labels_pred = self.surrogate_head(sent_embeddings)

        # Surrogate loss
        surr_loss = self.surr_loss(
            next_sent_labels_pred,
            next_sent_labels.to(dtype=torch.long)
        )

        if surr_loss_only:
            return {'loss': surr_loss}

        # Get the triplets
        triplet_indices = self.miner(
            sent_embeddings,
            sent_labels.to(dtype=torch.long)
        )

        # Semantic contrast loss
        sem_loss = self.sem_loss(sent_embeddings, indices_tuple=triplet_indices)


        return {
            'loss': sem_loss + surr_loss,
            'surr_loss': surr_loss
        }

    def training_step(self, batch, batch_idx, optimizer_idx):
        if optimizer_idx == 0:
            loss_dict = self.common_step(batch, batch_idx)
            # logs metrics for each training_step,
            # and the average across the epoch
            for k,v in loss_dict.items():
                self.log("tr_" + k, v.item(), prog_bar=True)

        else:
            loss_dict = self.common_step(batch, batch_idx, surr_loss_only=True)

        return loss_dict['loss']

    def validation_step(self, batch, batch_idx):
        loss_dict = self.common_step(batch, batch_idx)
        # logs metrics for each training_step,
        # and the average across the epoch
        for k,v in loss_dict.items():
            self.log("val_" + k, v.item(), prog_bar=True)

        return loss_dict['loss']

    def configure_optimizers(self):
        return [
            torch.optim.AdamW(
                self.embedder.parameters(),
                lr=self.embedder_lr,
                weight_decay=self.weight_decay
            ),
            torch.optim.AdamW(
                self.surrogate_head.parameters(),
                lr=self.surrogate_lr,
                weight_decay=self.weight_decay
            )
        ]


class SentenceEmbedder(pl.LightningModule):
    def __init__(
        self,
        mpath: str,
        triplet_margin: float = 0.5,
        embedder_lr: float = 1e-5,
        weight_decay: float = 1e-4
    ):
        """
        Args:
            mpath (str): Path to the `transformer` artifact files.
            num_classes (int): Number of classes in the data.
            triplet_margin (float): The margin to be used in triplet loss.
            embedder_lr (float, optional): Embedder's learning rate.
            Defaults to 1e-5.
            weight_decay (float, optional): Common weight decay co-efficient.
            Defaults to 1e-4.
        """
        super().__init__()
        self.embedder_lr = embedder_lr
        self.weight_decay = weight_decay

        # Base model        
        self.embedder = AutoModel.from_pretrained(mpath)
        # Pooling layer to get the sentence embedding
        self.pooling = Pooling(
            self.embedder.config.hidden_size,
            pooling_mode='mean'
        )

        # To mine the triplets
        self.miner = miners.BatchHardMiner()
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
        sent_tokens, (sent_labels, _) = batch
        sent_embeddings = self(**sent_tokens)

        # Get the triplets
        triplet_indices = self.miner(sent_embeddings, sent_labels)

        return self.sem_loss(sent_embeddings, indices_tuple=triplet_indices)

    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("tr_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("val_loss", loss, prog_bar=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.embedder.parameters(),
            lr=self.embedder_lr,
            weight_decay=self.weight_decay
        )
