from dataset import warm_image_cache
import torch

def training_loop(dataloader,
                model,
                device,
                loss_fn,
                optimizer,
                accuracy_tracking: None | list = None,
                loss_tracking: None | list = None,
                reporting: bool = False):
    '''

    '''

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
                print(f"BATCH {batch} -> Loss {loss:7f} [{current:>5f}/{size:>5d}]")

    if accuracy_tracking is not None:
        accuracy = (num_correct / num_seen).item()
        accuracy_tracking.append(accuracy)

    if loss_tracking is not None:
        avg_loss = (total_loss / num_seen).item()
        loss_tracking.append(avg_loss)

def validation_loop(dataloader,
                    model,
                    loss_fn,
                    device,
                    accuracy_tracking: None | list = None,
                    loss_tracking: None | list = None,
                    reporting: bool = False):
    '''

    '''

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
                    print(f"BATCH {batch} -> loss {loss:7f}  [{current:>5f}/{size:>5d}]")

    if accuracy_tracking is not None:
        accuracy = (num_correct / num_seen).item()
        accuracy_tracking.append(accuracy)

    if loss_tracking is not None:
        avg_loss = (total_loss / num_seen).item()
        loss_tracking.append(avg_loss)

def train_model(train_dataloader,
                val_dataloader,
                model,
                loss_fn,
                optimizer,
                device,
                checkpoint_name:str,
                image_directory: str,
                max_epochs: int = 15,
                patience: int = 3,
                min_delta:float|int = 1e-4,
                reporting: bool = False,
                early_stop:bool = True):

    """

    """

    train_loss_tracking = []
    train_accuracy_tracking = []
    val_loss_tracking = []
    val_accuracy_tracking = []

    best_val_loss = float('inf')
    epochs_since_improvement = 0

    warm_image_cache(image_directory)
    for epoch in range(max_epochs):
        print(f"Epoch {epoch + 1}...")

        training_loop(train_dataloader, model, device,
                      loss_fn, optimizer, train_accuracy_tracking,
                      train_loss_tracking, reporting)

        print("Validation...")

        validation_loop(val_dataloader, model, loss_fn,
                        device, val_accuracy_tracking,
                        val_loss_tracking, reporting)

        if early_stop:
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

            if epochs_since_improvement >= patience:
                print("Early stopping")
                break

def main():
    ...

if __name__ == '__main__':
    main()