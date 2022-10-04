from torch import optim
import torch
import torch.nn.functional as F
import json

from ser.model import Net
from dataclasses import dataclass, asdict



def train(run_path, params, train_dataloader, val_dataloader, val_dataloader_flipped, device, plotter):
    # setup model
    model = Net().to(device)

    # setup params
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

    # train
    for epoch in range(params.epochs):
        _train_batch(model, train_dataloader, optimizer, epoch, device, plotter)
        unflip_accuracy = _val_batch(model, val_dataloader, device, epoch, plotter, False)
        flip_accuracy = _val_batch(model, val_dataloader_flipped, device, epoch, plotter, True)
    
    accs = {"unflip": unflip_accuracy, "flip": flip_accuracy, "average": (unflip_accuracy+flip_accuracy)/2}

    # save model and save model params
    torch.save(model, run_path / "model.pt")
    with open(run_path / "Accuracies.json", "w") as f:
        json.dump(accs, f, indent=2)


def _train_batch(model, dataloader, optimizer, epoch, device, plotter):
    for i, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)
        model.train()
        optimizer.zero_grad()
        output = model(images)
        loss = F.nll_loss(output, labels)
        loss.backward()
        optimizer.step()
        print(
            f"Train Epoch: {epoch} | Batch: {i}/{len(dataloader)} "
            f"| Loss: {loss.item():.4f}"
        )
        plotter.plot('loss', 'train', 'Class Loss', i*(epoch+1), loss.item())


@torch.no_grad()
def _val_batch(model, dataloader, device, epoch, plotter, flipped):
    val_loss = 0
    correct = 0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        model.eval()
        output = model(images)
        val_loss += F.nll_loss(output, labels, reduction="sum").item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(labels.view_as(pred)).sum().item()
    val_loss /= len(dataloader.dataset)
    accuracy = correct / len(dataloader.dataset)
    if flipped: 
        title = "Flipped images: Class Accuracy"
        val = "Flipped val"
    else: 
        title = "Unflipped images: Class Accuracy"
        val = "Unflipped val"
    print(f"Val Epoch: {epoch} | Avg Loss: {val_loss:.4f} | Accuracy: {accuracy}")
    plotter.plot('loss', val, 'Class Loss', epoch, val_loss)
    plotter.plot('acc', val, title, epoch, accuracy)
    return accuracy

