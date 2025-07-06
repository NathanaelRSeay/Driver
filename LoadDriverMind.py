
import torch


def save_model(model, optimizer, epoch, loss, path="checkpoint.pth"):
    """
    Save model and optimizer state to a file.
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, path)
    print(f"✅ Model saved to: {path}")

def load_model(model, optimizer, path="checkpoint.pth", device='cpu'):
    """
    Load model and optimizer state from a file.
    Returns: model, optimizer, start_epoch, loss
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"✅ Model loaded from: {path} (Epoch {start_epoch})")
    return model.to(device), optimizer, start_epoch, loss