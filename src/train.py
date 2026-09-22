from pathlib import Path
import torch

from src.dataset import warm_image_cache

def training_loop(dataloader: torch.utils.data.DataLoader,
                model: torch.nn.Module,
                device: torch.device,
                loss_fn: torch.nn.Module,
                optimizer: torch.optim.Optimizer,
                accuracy_tracking: None | list = None,
                loss_tracking: None | list = None,
                reporting: bool = False):
    """
    Runs one epoch of training.

    Parameters
    ----------
        dataloader: torch.utils.data.DataLoader
            A DataLoader object that contains training data.

        model: torch.nn.Module
            A ResNet-18 model to be trained.

        device: torch.device
            A torch device to be used for training.

        loss_fn: torch.nn.Module
            A torch.nn.Module loss function to be used for training. The method allows custom methods, but correct and
            consistent methods are assumed.

        optimizer: torch.optim.Optimizer
            A torch optimizer to be used for training. As with the loss function, this method allows custom optimizers
            but assumes all optimizers are correctly implemented.

        accuracy_tracking: list or None
            If not None, a list to hold the average accuracy per epoch. If parameter is None, accuracy tracking will
            be disabled.

        loss_tracking: list or None
            If not None, a list to hold the average loss produced by the loss function. If parameter is None,
            loss tracking will be disabled.

        reporting: bool
            If True, method will report the loss value every 100 batches.
    """
    size = len(dataloader.dataset)
    model.train()
    total_loss = torch.zeros((), device=device)
    num_correct = torch.zeros((), device=device)
    num_seen = 0

    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.detach() * y.size(0)
        num_correct += (pred.argmax(dim=1) == y).sum()
        num_seen += y.size(0)

        if reporting:
            if batch % 100 == 0:
                loss, current = (loss.item(),
                                 min((batch + 1) * dataloader.batch_size, size))
                print(f"BATCH {batch} -> Loss {loss:7f} [{current:>5d}/{size:>5d}]")

    if accuracy_tracking is not None:
        accuracy = (num_correct / num_seen).item()
        accuracy_tracking.append(accuracy)

    if loss_tracking is not None:
        avg_loss = (total_loss / num_seen).item()
        loss_tracking.append(avg_loss)

def validation_loop(dataloader: torch.utils.data.DataLoader,
                    model: torch.nn.Module,
                    device: torch.device,
                    loss_fn: torch.nn.Module,
                    accuracy_tracking: None | list = None,
                    loss_tracking: None | list = None,
                    reporting: bool = False):
    """
    Runs one epoch of validation.

    Parameters
    ----------
        dataloader: torch.utils.data.DataLoader
             A DataLoader object that contains validation data.

        model: torch.nn.Module
            A ResNet-18 model to be validation.

        device: torch.device
            A torch device to be used for validation.

        loss_fn: torch.nn.Module
            A torch.nn.Module loss function to be used for validation. The method allows custom methods, but correct and
            consistent methods are assumed.

        accuracy_tracking: list or None
            If not None, a list to hold the average accuracy per epoch. If parameter is None, accuracy tracking will
            be disabled.

        loss_tracking: list or None
            If not None, a list to hold the average loss produced by the loss function. If parameter is None,
            loss tracking will be disabled.

        reporting: bool
            If True, method will report the loss value every 100 batches.
        """

    size = len(dataloader.dataset)
    model.eval()
    total_loss = torch.zeros((), device=device)
    num_correct = torch.zeros((), device=device)
    num_seen = 0

    with (torch.no_grad()):
        for batch, (X, y) in enumerate(dataloader):
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            pred = model(X)
            loss = loss_fn(pred, y)

            total_loss += loss.detach() * y.size(0)
            num_correct += (pred.argmax(dim=1) == y).sum()
            num_seen += y.size(0)

            if reporting:
                if batch % 100 == 0:
                    loss, current = (loss.item(),
                                    min((batch + 1) * dataloader.batch_size, size))
                    print(f"BATCH {batch} -> loss {loss:7f}  [{current:>5d}/{size:>5d}]")

    if accuracy_tracking is not None:
        accuracy = (num_correct / num_seen).item()
        accuracy_tracking.append(accuracy)

    if loss_tracking is not None:
        avg_loss = (total_loss / num_seen).item()
        loss_tracking.append(avg_loss)

def train_model(train_dataloader: torch.utils.data.DataLoader,
                val_dataloader: torch.utils.data.DataLoader,
                model: torch.nn.Module,
                loss_fn: torch.nn.Module,
                optimizer: torch.optim.Optimizer,
                device: torch.device,
                checkpoint_name:str,
                image_directory: str | Path | None = None,
                max_epochs: int = 15,
                patience: int | None= 3,
                min_delta:float | int = 1e-4,
                reporting: bool = False,
                early_stop:bool = True,
                warm_cache: bool = False):
    """
    Trains a ResNet-18 model based on a predefined number of epochs.

    Parameters
    ----------
        train_dataloader: torch.utils.data.DataLoader
            A DataLoader object that contains training data.

        val_dataloader: torch.utils.data.Dataloader
            A DataLoader object that contains the validation data.

        model: torch.nn.Module
            The ResNet-18 model to be trained.

        loss_fn: torch.nn.Module
            A torch.nn.Module loss function to be used for training. The method allows custom methods, but correct and
            consistent methods are assumed.

        optimizer: torch.optim.Optimizer
            A torch optimizer to be used for training. As with the loss function, this method allows custom optimizers
            but assumes all optimizers are correctly implemented.

        device: torch.device
            A torch device to be used for training and validation.

        checkpoint_name: str
            A string containing the desired name of the saved model after training is completed.

        image_directory: str, Path, or None
            If not None, the string or Path object to the directory containing the images. If warm_cache is True, an
            image directory is expected to be passed.

        max_epochs: int
            Maximum number of epochs to train for.

        patience: int | None
            An int representing how many epochs the model will allow to pass before early stopping. If early_stop is
            True, a value for patience is expected

        min_delta: float | int
            A float or int specifying minimum change expected from each epoch in order to be defined as a
            significant change.

        reporting: bool
            If True, both training and validation will be reported every 100 batches during each epoch.

        early_stop: bool
            If True, the method will perform an early stop if the model doesn't change significantly within the
            specified number of epochs defined by the patience parameter.

        warm_cache: bool
            If True, the provided image directory will be warmed before the first training loop.

    Returns
    -------
        return_data: dict
            A dictionary containing the training loss, training accuracy, validation loss, and validation accuracy.

    """

    if (patience is None) and (early_stop == True):
        raise ValueError('patience must be defined if early_stop is set to True.')

    train_loss_tracking = []
    train_accuracy_tracking = []
    val_loss_tracking = []
    val_accuracy_tracking = []

    best_val_loss = float('inf')
    epochs_since_improvement = 0

    if warm_cache:
        if image_directory is not None:
            warm_image_cache(image_directory)
        else:
            raise ValueError('image_directory must be provided since warm_cache was set to True.')

    for epoch in range(max_epochs):
        print(f"Epoch {epoch + 1}...")

        training_loop(train_dataloader, model, device,
                      loss_fn, optimizer, train_accuracy_tracking,
                      train_loss_tracking, reporting)

        print("Validation...")

        validation_loop(val_dataloader, model, device, loss_fn, val_accuracy_tracking,
                        val_loss_tracking, reporting)

        if val_loss_tracking[-1] < best_val_loss - min_delta:
            best_val_loss = val_loss_tracking[-1]
            epochs_since_improvement = 0

            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                "val_loss": val_loss_tracking[-1],
                "val_accuracy": val_accuracy_tracking[-1]
            }, checkpoint_name)

        else:
            epochs_since_improvement += 1

        if early_stop:
            if epochs_since_improvement >= patience:
                print("Early stopping")
                break

    return_data = {"train_loss": train_loss_tracking,
                   "train_accuracy": train_accuracy_tracking,
                   "val_loss": val_loss_tracking,
                   "val_accuracy": val_accuracy_tracking}

    return return_data