import torch

import dataset

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
                checkpoint_name:str|None = None,
                max_epochs: int = 15,
                patience: int = 3,
                min_delta:float|int = 1e-4,
                train_accuracy_tracking: None | list = None,
                train_loss_tracking: None | list = None,
                val_accuracy_tracking: None | list = None,
                val_loss_tracking: None | list = None,
                reporting: bool = False,
                early_stop:bool = True):

    

def main():
    ...

if __name__ == '__main__':
    main()