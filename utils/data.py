import logging
import os
from collections import Counter

import numpy as np
from torchvision import datasets, transforms
from utils.toolkit import split_images_labels
from sklearn.model_selection import train_test_split


class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None


class iCXRDisesases10(iData):
    use_path = True  # Images are stored in paths

    # train_trsf = [
    #     transforms.Grayscale(num_output_channels=3),
    #     transforms.Resize((32, 32)),
    #     transforms.RandomCrop(32, padding=4),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ColorJitter(brightness=63 / 255),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)),
    # ]

    # test_trsf = [
    #     transforms.Grayscale(num_output_channels=3),
    #     transforms.Resize((32, 32)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)),
    # ]

    # common_trsf = [
    #     transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)),
    # ]

    image_dim = 32
    train_trsf = [
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((image_dim, image_dim)),  # Larger size
        transforms.RandomCrop(image_dim, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),  # Slight rotation for variability
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),  # Small translations
        transforms.RandomResizedCrop(image_dim, scale=(0.8, 1.0)),  # Crop with scaling
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761))
    ]
    test_trsf = [
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((image_dim, image_dim)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761))
    ]
    common_trsf = []
    
    def download_data(self):
        # folder_path = "/content/drive/MyDrive/nih-chest-xrays-filtered"
        folder_path = "/kaggle/input/nih-chest-xrays-filtered/nih-chest-xrays-filtered"
        logging.info("Starting to load COVID dataset from folder structure...")

        class_order = [
            "Covid19-CovidX", "Healthy-CovidX", "Pneumonia-CovidX",
            "Effusion", "Pneumothorax", "Cardiomegaly",
            "Atelectasis", "Consolidation", "Nodule", "Infiltration"
        ]
        label_map = {class_name: idx for idx, class_name in enumerate(class_order)}
        logging.info(f"Label Map: {label_map}")

        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif", ".webp"}

        image_paths = []
        labels = []

        for class_name in class_order:
            class_path = os.path.join(folder_path, class_name)
            if not os.path.isdir(class_path):
                logging.warning(f"Skipping missing class folder: {class_name}")
                continue

            class_images = [
                os.path.join(class_path, img) for img in os.listdir(class_path)
                if os.path.isfile(os.path.join(class_path, img))
                   and os.path.splitext(img)[1].lower() in image_extensions
            ]

            logging.info(f"Loaded {len(class_images)} valid images for class {class_name}")

            image_paths.extend(class_images)
            labels.extend([label_map[class_name]] * len(class_images))

        image_paths = np.array(image_paths)
        labels = np.array(labels)

        total_class_counts = Counter(labels)
        for class_label, count in total_class_counts.items():
            logging.info(f"Total images for class {class_label} ({class_order[class_label]}): {count}")

        min_samples_per_class = min(total_class_counts.values())
        test_size_per_class = max(1, int(min_samples_per_class * 0.2))

        train_paths, test_paths, train_labels, test_labels = train_test_split(
            image_paths, labels, test_size=test_size_per_class * len(class_order),
            stratify=labels, random_state=42
        )

        train_counts = Counter(train_labels)
        test_counts = Counter(test_labels)

        for class_label in total_class_counts.keys():
            train_count = train_counts.get(class_label, 0)
            test_count = test_counts.get(class_label, 0)
            logging.info(f"Train images for class {class_label} ({class_order[class_label]}): {train_count}")
            logging.info(f"Test images for class {class_label} ({class_order[class_label]}): {test_count}")

        # ✅ **Store train and test data separately (Old Format)**
        self.train_data = train_paths  # File paths only
        self.train_targets = train_labels  # Labels only
        self.test_data = test_paths  # File paths only
        self.test_targets = test_labels  # Labels only

        logging.info(f"Total train images: {len(self.train_data)}")
        logging.info(f"Total test images: {len(self.test_data)}")

    class_order = list(range(10))  # Follow the defined order


class iCIFAR10(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=63 / 255),
        transforms.ToTensor(),
    ]
    test_trsf = [transforms.ToTensor()]
    common_trsf = [
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)
        ),
    ]

    class_order = np.arange(10).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR10("./data", train=True, download=True)
        test_dataset = datasets.cifar.CIFAR10("./data", train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )


class iCIFAR100(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
        transforms.ToTensor()
    ]
    test_trsf = [transforms.ToTensor()]
    common_trsf = [
        transforms.Normalize(
            mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)
        ),
    ]

    class_order = np.arange(100).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR100("./data", train=True, download=True)
        test_dataset = datasets.cifar.CIFAR100("./data", train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )


class iImageNet1000(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(1000).tolist()

    def download_data(self):
        assert 0, "You should specify the folder of your dataset"
        train_dir = "[DATA-PATH]/train/"
        test_dir = "[DATA-PATH]/val/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iImageNet100(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(1000).tolist()

    def download_data(self):
        assert 0, "You should specify the folder of your dataset"
        train_dir = "[DATA-PATH]/train/"
        test_dir = "[DATA-PATH]/val/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)
