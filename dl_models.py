import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Tuple


class NNAbstractClass(nn.Module, ABC):
    def __init__(
        self,
        input_embedding_size: torch.Tensor,
        hidden_size: int,
        output_size: int
    ) -> None:
        """
        Parameters
        ----------
        input_embedding_size : torch.Tensor
            Input embedding vector for a single example.
        hidden_size : int
            Size of the hidden state.
        output_size : int
            Number of output classes.
        """
        super().__init__()
        self.input_embedding_size = input_embedding_size
        self.input_size = input_embedding_size.size(0)
        self.hidden_size = hidden_size
        self.output_size = output_size

    def train(
        self,
        target: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module
    ) -> Tuple[torch.Tensor, float]:
        """
        Template method defining one training step.

        Parameters
        ----------
        target : torch.Tensor
            Ground truth label tensor.
        optimizer : torch.optim.Optimizer
            Optimizer used for updating model parameters.
        criterion : nn.Module
            Loss function.

        Returns
        -------
        output : torch.Tensor
            Model prediction.
        loss : float
            Scalar loss value.
        """
        x = self.resize_input()
        h = self.resize_hidden()
        h_next, output = self.forward(x, h)

        loss = self.compute_loss(output, target, criterion)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return output, loss.item()

    def resize_input(self) -> torch.Tensor:
        """
        Returns
        -------
        torch.Tensor
            Input tensor reshaped to batch dimension.
        """
        return self.input_embedding_size.unsqueeze(0)

    def resize_hidden(self) -> torch.Tensor:
        """
        Returns
        -------
        torch.Tensor
            Initialized hidden state tensor.
        """
        return torch.zeros(1, self.hidden_size)

    def compute_loss(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        criterion: nn.Module
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        output : torch.Tensor
            Model predictions.
        target : torch.Tensor
            Ground truth labels.
        criterion : nn.Module
            Loss function.

        Returns
        -------
        torch.Tensor
            Computed loss.
        """
        return criterion(output, target)

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        h : torch.Tensor
            Hidden state tensor.

        Returns
        -------
        hidden : torch.Tensor
            Next hidden state.
        output : torch.Tensor
            Model output.
        """
        pass


class PyTorchRNN(NNAbstractClass):
    def __init__(
        self,
        input_embedding_size: torch.Tensor,
        hidden_size: int,
        output_size: int
    ) -> None:
        """
        Parameters
        ----------
        input_embedding_size : torch.Tensor
            Input embedding vector.
        hidden_size : int
            Hidden layer size.
        output_size : int
            Number of target classes.
        """
        super().__init__(input_embedding_size, hidden_size, output_size)

        self.input_hidden_to_hidden = nn.Linear(
            self.input_size + hidden_size, hidden_size
        )
        self.input_hidden_to_output = nn.Linear(
            self.input_size + hidden_size, output_size
        )
        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        h : torch.Tensor
            Hidden state tensor.

        Returns
        -------
        hidden : torch.Tensor
            Updated hidden state.
        output : torch.Tensor
            Model prediction after activation.
        """
        x_to_h = torch.cat((x, h), dim=1)
        hidden = self.input_hidden_to_hidden(x_to_h)
        y_hat = self.input_hidden_to_output(x_to_h)
        output = self.sigmoid(y_hat)
        return hidden, output