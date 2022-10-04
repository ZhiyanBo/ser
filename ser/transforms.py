from torchvision import transforms as torch_transforms


def transforms(*stages, prob = 0):
    transforms_list = [torch_transforms.ToTensor()]
    for i in stages:
        if i == flip: 
            transforms_list.append(flip(prob))
        else:
            transforms_list.append(i())
    return torch_transforms.Compose(transforms_list)


def normalize():
    """
    Normalize a tensor to have a mean of 0.5 and a std dev of 0.5
    """
    return torch_transforms.Normalize((0.5,), (0.5,))


def flip(prob):
    """
    Flip a tensor both vertically and horizontally
    """
    return torch_transforms.Compose(
        [
            torch_transforms.RandomHorizontalFlip(p=prob),
            torch_transforms.RandomVerticalFlip(p=prob),
        ]
    )