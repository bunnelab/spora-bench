import copy

import numpy as np
import torch
from einops import rearrange
from loguru import logger
from torch import nn


class MultiHeadGatedABMIL(nn.Module):

    def __init__(self, 
                 emb_dim: int, 
                 hidden_dim: int, 
                 num_heads: int = 1,
                 n_branches: int = 8,
                 classifier: torch.nn.Module = None,
                 dropout: float = 0.0
                 ) -> None:
        """
        Implements a multi-head ABMIL model with gated attention.
        Args:
            emb_dim: dimension of input embeddings
            hidden_dim: dimension of hidden layer in attention
            num_heads: number of attention heads
            n_branches: number of branches in the classifier
            classifier: classifier module
            dropout: dropout rate
        """
        super().__init__()

        self.V = nn.ModuleList([nn.Sequential(nn.Linear(emb_dim, hidden_dim), nn.Dropout(dropout)) for _ in range(num_heads)])
        self.U = nn.ModuleList([nn.Sequential(nn.Linear(emb_dim, hidden_dim), nn.Dropout(dropout)) for _ in range(num_heads)])
        self.W = nn.ModuleList([nn.Linear(hidden_dim, n_branches) for _ in range(num_heads)])
        
        self.num_heads = num_heads
        self.n_branches = n_branches

        if classifier is not None:
            self.classifier = classifier
        else:
            self.classifier = nn.Identity()

    def forward(self,
                x: torch.Tensor, 
                mask: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): input of size B x S x D
            mask (torch.Tensor, optional): mask of size B x S indicating padding (0). Defaults to None.
        Returns:
            torch.Tensor: output of size B x num_classes after classifier
            torch.Tensor: output of size B x (num_branches * hidden_dim * num_heads) before classifier
        """
        outputs = []

        for i in range(self.num_heads):
            v_x = self.V[i](x)
            u_x = self.U[i](x)

            v_x = nn.functional.tanh(v_x)
            u_x = nn.functional.sigmoid(u_x)

            h = v_x * u_x

            attn_scores = self.W[i](h) #B x S x H
            if mask is not None:
                attn_scores = attn_scores.masked_fill(mask.unsqueeze(2), -1e9)
            attn_weights = nn.functional.softmax(attn_scores, dim=1) # B x S x H

            output = torch.einsum('bsh,bsd->bhd', attn_weights, x) # B x H x D
            outputs.append(output)
        
        output = torch.cat(outputs, dim=-1)
        output_flat = output.reshape(-1, x.size(2) * self.n_branches * self.num_heads)

        return self.classifier(output_flat), output_flat

    def compute_attention(self,
                          x: torch.Tensor, 
                          mask: torch.Tensor | None = None,
                          ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input of size B x S x D
            mask (torch.Tensor, optional): mask of size B x S indicating padding (0). Defaults to None.
        Returns:
                torch.Tensor: attention weights of size B x n_branches x num_heads x S
        """
        outputs = []
        attentions = []

        for i in range(self.num_heads):
            v_x = self.V[i](x)
            u_x = self.U[i](x)

            v_x = nn.functional.tanh(v_x)
            u_x = nn.functional.sigmoid(u_x)

            h = v_x * u_x

            attn_scores = self.W[i](h) #B x S x H
            if mask is not None:
                attn_scores = attn_scores.masked_fill(mask.unsqueeze(2), -1e9)
            attn_weights = nn.functional.softmax(attn_scores, dim=1) # B x S x H

            output = torch.einsum('bsh,bsd->bhd', attn_weights, x) # B x H x D
            outputs.append(output)
            attentions.append(attn_weights)
        
        output = torch.cat(outputs, dim=1)
        attentions = torch.stack(attentions, dim=-1) # Shape: batch_size x num_images x n_branches x n_heads
        attentions = rearrange(attentions, 'b n r h -> b r h n') # Shape: batch_size x n_branches x n_heads x num_images

        return attentions
        

class GatedABMILClassifierWithValidation(nn.Module):
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int, 
                 num_heads: int, 
                 num_branches: int, 
                 num_classes: int, 
                 patience: int = 5, 
                 dropout: float = 0, 
                 monitor: str = "valid_loss"
                 ) -> None:
        super().__init__()

        assert monitor in ["valid_loss", "valid_accuracy"], "monitor must be either valid_loss or valid_accuracy"
        self.num_classes = num_classes
        classifier = nn.Sequential(
            nn.Linear(num_branches * input_dim * num_heads, self.num_classes), 
        )

        self.patience = patience
        self.patience_counter = 0
        self.model = MultiHeadGatedABMIL(input_dim, hidden_dim, num_heads=num_heads, n_branches=num_branches, dropout=dropout, classifier=classifier)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.monitor = monitor

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        return self.model(x, mask=mask)

    def compute_attention(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        return self.model.compute_attention(x, mask=mask)
    
    @torch.no_grad()
    def valid_eval(self, 
                   valid_dl: torch.utils.data.DataLoader, 
                   loss_fn: torch.nn.Module
                   ) -> tuple[float, float]:
        """Evaluates the model on the validation set and returns the average loss and accuracy.
        Args:
            valid_dl (torch.utils.data.DataLoader): validation data loader
            loss_fn (torch.nn.Module): loss function
        Returns:
            tuple[float, float]: average loss and accuracy
        """
        self.eval()
        valid_loss = 0
        predictions = []
        ground_truth = []
        for batch in valid_dl:
            bags, masks, labels = batch
            bags = bags.to(self.device)
            masks = masks.to(self.device)
            labels = labels.to(self.device)
            logits, _ = self.forward(bags, mask=masks)

            loss = loss_fn(logits, labels)
            preds = torch.argmax(logits, dim=1)
            
            predictions.append(preds.cpu().numpy())
            ground_truth.append(labels.cpu().numpy())
            valid_loss += loss.item()

        predictions = np.concatenate(predictions)
        ground_truth = np.concatenate(ground_truth)
        accuracy = (predictions == ground_truth).mean()
        
        return valid_loss / len(valid_dl), accuracy
    

    def train_model(self, 
                    train_dl: torch.utils.data.DataLoader, 
                    valid_dl: torch.utils.data.DataLoader, 
                    num_epochs: int, 
                    optimizer: torch.optim.Optimizer, 
                    loss_fn: torch.nn.Module, 
                    test_dl: torch.utils.data.DataLoader | None = None
                    ) -> torch.nn.Module:
        """
        Trains the model. train_dl should return batches of the form (bags, masks, labels) where bags BxSxD, masks BxS, labels B
        Args:            
            train_dl (torch.utils.data.DataLoader): training data loader
            valid_dl (torch.utils.data.DataLoader): validation data loader
            num_epochs (int): number of epochs to train for
            optimizer (torch.optim.Optimizer): optimizer to use for training
            loss_fn (torch.nn.Module): loss function to use for training
            test_dl (torch.utils.data.DataLoader, optional): test data loader for evaluating test performance after each epoch. Defaults to None.
        Returns:
            torch.nn.Module: the trained model with the best validation performance
        """
        self.to(self.device)
        self.average_train_loss = []
        self.average_train_accuracy = []
        self.average_valid_loss = []
        self.average_valid_accuracy = []

        if test_dl is not None:
            self.average_test_loss = []
            self.average_test_accuracy = []
        
        self.best_val_acc = 0
        self.best_val_loss = np.inf
        best_model = copy.deepcopy(self.model.state_dict())

        for ep in (range(num_epochs)):
            train_loss = 0
            self.train()
            for batch in train_dl:
                bags, masks, labels = batch
                bags = bags.to(self.device, non_blocking=True)
                masks = masks.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                optimizer.zero_grad()
                logits, _ = self.forward(bags, mask=masks)
                loss = loss_fn(logits, labels)

                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            self.average_train_loss.append(train_loss / len(train_dl))
            valid_loss, valid_accuracy = self.valid_eval(valid_dl, loss_fn)
            self.average_valid_accuracy.append(valid_accuracy)
            self.average_valid_loss.append(valid_loss)

            logger.info(f"Epoch: {ep} Train Loss: {train_loss / len(train_dl):.3f}, Valid Loss: {valid_loss:.3f}, Valid Accuracy: {valid_accuracy:.3f}")
                
            if (self.monitor == "valid_accuracy" and valid_accuracy > self.best_val_acc) \
                or (self.monitor == "valid_loss" and valid_loss < self.best_val_loss):
                self.best_val_acc = valid_accuracy
                self.best_val_loss = valid_loss
                best_model = copy.deepcopy(self.model.state_dict())
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            if self.patience_counter >= self.patience:
                logger.info(f"Early stopping with best {self.monitor} @ loss: {self.best_val_loss:.3f}, acc: {self.best_val_acc:.3f}")
                self.model.load_state_dict(best_model)
                return self.model

        self.model.load_state_dict(best_model)
        return self.model


    @torch.no_grad()
    def compute_predictions_and_truth(self, dataloader, loss_fn):
        """
        dataloader should return batches of the form (bags, masks, labels) where bags BxSxD, masks BxS, labels B
        Returns predictions and ground truth as numpy 1D arrays
        """
        self.to(self.device)
        self.eval()
        predictions = []
        ground_truth = []

        test_loss = 0
        for batch in dataloader:
            bags, masks, labels = batch
            bags = bags.to(self.device)
            masks = masks.to(self.device)
            labels = labels.to(self.device)
            logits, _ = self.forward(bags, mask=masks)
            loss = loss_fn(logits, labels)
            preds = torch.argmax(logits, dim=1)
            
            predictions.append(preds.cpu().numpy())
            ground_truth.append(labels.cpu().numpy())

            test_loss += loss.item()
            
        predictions = np.concatenate(predictions)
        ground_truth = np.concatenate(ground_truth)

        return predictions, ground_truth, test_loss / len(dataloader)