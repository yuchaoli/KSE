#coding: utf-8
import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data


def load_data(train_batch_size, test_batch_size):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    test_trainsform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=train_batch_size,
                                              shuffle=True, num_workers=2)

    test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=test_trainsform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=test_batch_size,
                                             shuffle=False, num_workers=2)

    return train_loader, test_loader