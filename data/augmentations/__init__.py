from torchvision import transforms
from data.augmentations.cut_out import *
from data.augmentations.randaugment import RandAugment

def get_transform(transform_type='default', image_size=32, args=None):

    if transform_type == 'default':

        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        test_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    elif transform_type == 'pytorch-cifar':

        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)

        train_transform = transforms.Compose([
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        test_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    elif transform_type == 'ARPL':

        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        test_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    elif transform_type == 'cgnl':

        base_size = int((512 / 448) * image_size)

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(
                size=image_size, scale=(0.08, 1.25)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        test_transform = transforms.Compose([
            transforms.Resize(base_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    elif transform_type == 'cutout':

        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.2470, 0.2435, 0.2616])

        train_transform = transforms.Compose([
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            normalize(mean, std),
            cutout(mask_size=int(image_size / 2),
                   p=1,
                   cutout_inside=False),
            to_tensor(),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    elif transform_type == 'rand-augment':

        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)

        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        train_transform.transforms.insert(0, RandAugment(1, 9, args=args))

        test_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    elif transform_type == 'openhybrid':

        train_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((image_size, image_size)),
            transforms.RandomCrop(image_size, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        test_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    else:

        raise NotImplementedError

    return (train_transform, test_transform)