from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from src.sequence_dataset import DeepfakeSequenceDataset

def get_transforms(config=None):
    train_transform = transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.GaussianBlur(3)
        ], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])

    return train_transform, eval_transform

def get_dataloaders(config=None):
    train_transform, eval_transform = get_transforms(config=config)

    train_dataset = datasets.ImageFolder(root=config.TRAIN_DIR, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=config.VAL_DIR, transform=eval_transform)
    test_dataset = datasets.ImageFolder(root=config.TEST_DIR, transform=eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=config.Batch_size, shuffle=True, num_workers=config.Num_workers, pin_memory=config.pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=config.Batch_size, shuffle=False, num_workers=config.Num_workers, pin_memory=config.pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=config.Batch_size, shuffle=False, num_workers=config.Num_workers, pin_memory=config.pin_memory)

    return train_loader, val_loader, test_loader

def get_sequence_dataloader(config=None):
    train_transform, eval_transform = get_transforms(config=config)

    train_sequence_dataset = DeepfakeSequenceDataset(root_dir=config.TRAIN_DIR, transform=train_transform)
    val_sequence_dataset = DeepfakeSequenceDataset(root_dir=config.VAL_DIR, transform=eval_transform)
    test_sequence_dataset = DeepfakeSequenceDataset(root_dir=config.TEST_DIR, transform=eval_transform)

    frame, label = train_sequence_dataset[0]

    print(f"train seq examples: {frame.shape}, label: {label}")


    train_sequence_loader = DataLoader(train_sequence_dataset, batch_size=config.Batch_size_seq, shuffle=True, num_workers=config.Num_workers, pin_memory=config.pin_memory)
    val_sequence_loader = DataLoader(val_sequence_dataset, batch_size=config.Batch_size_seq, shuffle=False, num_workers=config.Num_workers, pin_memory=config.pin_memory)
    test_sequence_loader = DataLoader(test_sequence_dataset, batch_size=config.Batch_size_seq, shuffle=False, num_workers=config.Num_workers, pin_memory=config.pin_memory)

    return train_sequence_loader, val_sequence_loader, test_sequence_loader

