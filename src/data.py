from pathlib import Path
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

def get_loaders(batch_size=64, num_workers=0):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
        ),
    ])

    train_for_split = datasets.CIFAR100(
        root=str(DATA_DIR), train=True, transform=transform, download=True
    )
    train_indices, val_indices = random_split(
        range(len(train_for_split)),
        [45000, 5000],
        generator=__import__("torch").Generator().manual_seed(42),
    )

    val_dataset = datasets.CIFAR100(
        root=str(DATA_DIR), train=True, transform=transform, download=True
    )
    from torch.utils.data import Subset
    val_dataset = Subset(val_dataset, val_indices.indices)

    test_dataset = datasets.CIFAR100(
        root=str(DATA_DIR), train=False, transform=transform, download=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False,
    )
    return val_loader, test_loader
