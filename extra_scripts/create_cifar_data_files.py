import os
import json
import torchvision
import torchvision.transforms as transforms
import argparse

def get_argument_parser():
    """
    List of arguments supported by the script
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        choices=['cifar10', 'cifar100'],
        type=str,
        help="specify cifar10 or cifar100",
    )
    parser.add_argument("--classname", default='classes.json', type=str)

    return parser


if __name__ == '__main__':
    args = get_argument_parser().parse_args()

    classname_path = os.path.join("./configs", args.classname)
    with open(classname_path, 'r') as classname_file:
        classnames = json.load(classname_file)[args.dataset]

    # Define transformations to apply to the images
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert images to tensors
    ])

    # Download CIFAR-10/100 dataset
    if args.dataset == 'cifar10':
        train_dataset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=False, download=True, transform=transform)
    else:
        train_dataset = torchvision.datasets.CIFAR100(root='./data/cifar100', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR100(root='./data/cifar100', train=False, download=True, transform=transform)


    # Create train and test folders
    import os

    train_folder = f'./data/{args.dataset}/train'
    test_folder = f'./data/{args.dataset}/test'

    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    # Move images to train folder
    for i, (image, label) in enumerate(train_dataset):
        folder = os.path.join(train_folder, str(label))
        os.makedirs(folder, exist_ok=True)
        image_path = os.path.join(folder, f"{i}.png")
        torchvision.utils.save_image(image, image_path)

    # Move images to test folder
    for i, (image, label) in enumerate(test_dataset):
        folder = os.path.join(test_folder, str(label))
        os.makedirs(folder, exist_ok=True)
        image_path = os.path.join(folder, f"{i}.png")
        torchvision.utils.save_image(image, image_path)

    print("Dataset downloaded and saved in train and test folders.")

    def rename_folders(path, new_names):
        # Iterate over folder names 1 to 9
        for i in range(0, int(args.dataset[5:])):
            old_name = os.path.join(path, str(i))
            new_name = os.path.join(path, new_names[i])

            # Check if the old folder exists
            if os.path.exists(old_name):
                try:
                    os.rename(old_name, new_name)
                    print(f"Renamed {old_name} to {new_name}")
                except Exception as e:
                    print(f"Error renaming {old_name}: {e}")
            else:
                print(f"Folder {old_name} does not exist.")

    rename_folders(train_folder, classnames)
    rename_folders(test_folder, classnames)
