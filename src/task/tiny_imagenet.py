import torch
from torch.utils.data import DataLoader
import os
from src.dataloader.dataloader import iidLoader, dirichletLoader
from src.models import mobilenetv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import sys
from PIL import Image


def get_tinyimg_model(use_cuda=False):
    model = mobilenetv2.mobilenetv2()
    path = os.getcwd()
    pre_weights = torch.load(os.path.join(path, 'models/pretrained/mobilenetv2_1.0-0c6065bc.pth'))
    pre_dict = {k: v for k, v in pre_weights.items() if "classifier" not in k}
    model.load_state_dict(pre_dict, strict=False)
    # for param in model.features.parameters():
    #     param.requires_grad = False
    if use_cuda:
        model = model.cuda()
    return model


def get_tinyimg_data(root_dir, train=True):
    id_dict = {}
    for i, line in enumerate(open(root_dir + 'wnids.txt', 'r')):
        id_dict[line.replace('\n', '')] = i
    if train:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)
                )
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)
                )
            ]
        )
    if train:
        dataset = TinyImageNet(root=root_dir, train=True, transform=transform)
    else:
        dataset = TinyImageNet(root=root_dir, train=False, transform=transform)
    return dataset


def get_train_loader(root_dir, n_workers, alpha=1.0, batch_size=64, noniid=False):
    dataset = get_tinyimg_data(root_dir=root_dir, train=True)
    if not noniid:
        loader = iidLoader(size=n_workers, dataset=dataset, bsz=batch_size)
    else:
        loader = dirichletLoader(size=n_workers, dataset=dataset, alpha=alpha, bsz=batch_size)
    return loader


def get_test_loader(root_dir, batch_size):
    dataset = get_tinyimg_data(root_dir=root_dir, train=False)
    return torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)


# class TrainTinyImageNetDataset(Dataset):
#     def __init__(self, id, transform=None):
#         self.filenames = glob.glob("../../datasets/tiny-imagenet/train/*/*/*.JPEG")
#         self.transform = transform
#         self.id_dict = id
#
#     def __len__(self):
#         return len(self.filenames)
#
#     def __getitem__(self, idx):
#         img_path = self.filenames[idx]
#         data = read_image(img_path)
#         if data.shape[0] == 1:
#             data = read_image(img_path, ImageReadMode.RGB)
#         target = self.id_dict[img_path.split('/')[4]]
#         if self.transform:
#             data = self.transform(data.type(torch.FloatTensor))
#         return data, target
#
#
# class TestTinyImageNetDataset(Dataset):
#     def __init__(self, id, transform=None):
#         self.filenames = glob.glob("../../datasets/tiny-imagenet/val/images/*.JPEG")
#         self.transform = transform
#         self.id_dict = id
#         self.cls_dic = {}
#         for i, line in enumerate(open('../../datasets/tiny-imagenet/val/val_annotations.txt', 'r')):
#             a = line.split('\t')
#             img, cls_id = a[0], a[1]
#             self.cls_dic[img] = self.id_dict[cls_id]
#
#     def __len__(self):
#         return len(self.filenames)
#
#     def __getitem__(self, idx):
#         img_path = self.filenames[idx]
#         image = read_image(img_path)
#         if image.shape[0] == 1:
#             image = read_image(img_path, ImageReadMode.RGB)
#         label = self.cls_dic[img_path.split('/')[-1]]
#         if self.transform:
#             image = self.transform(image.type(torch.FloatTensor))
#         return image, label


class TinyImageNet(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.Train = train
        self.root_dir = root
        self.transform = transform
        self.train_dir = os.path.join(self.root_dir, "train")
        self.val_dir = os.path.join(self.root_dir, "val")

        if (self.Train):
            self._create_class_idx_dict_train()
        else:
            self._create_class_idx_dict_val()

        self._make_dataset(self.Train)

        words_file = os.path.join(self.root_dir, "words.txt")
        wnids_file = os.path.join(self.root_dir, "wnids.txt")

        self.set_nids = set()

        with open(wnids_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                self.set_nids.add(entry.strip("\n"))

        self.class_to_label = {}
        with open(words_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                words = entry.split("\t")
                if words[0] in self.set_nids:
                    self.class_to_label[words[0]] = (words[1].strip("\n").split(","))[0]

    def _create_class_idx_dict_train(self):
        if sys.version_info >= (3, 5):
            classes = [d.name for d in os.scandir(self.train_dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(self.train_dir) if os.path.isdir(os.path.join(self.train_dir, d))]
        classes = sorted(classes)
        num_images = 0
        for root, dirs, files in os.walk(self.train_dir):
            for f in files:
                if f.endswith(".JPEG"):
                    num_images = num_images + 1

        self.len_dataset = num_images;

        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}

    def _create_class_idx_dict_val(self):
        val_image_dir = os.path.join(self.val_dir, "images")
        if sys.version_info >= (3, 5):
            images = [d.name for d in os.scandir(val_image_dir) if d.is_file()]
        else:
            images = [d for d in os.listdir(val_image_dir) if os.path.isfile(os.path.join(self.train_dir, d))]
        val_annotations_file = os.path.join(self.val_dir, "val_annotations.txt")
        self.val_img_to_class = {}
        set_of_classes = set()
        with open(val_annotations_file, 'r') as fo:
            entry = fo.readlines()
            for data in entry:
                words = data.split("\t")
                self.val_img_to_class[words[0]] = words[1]
                set_of_classes.add(words[1])

        self.len_dataset = len(list(self.val_img_to_class.keys()))
        classes = sorted(list(set_of_classes))
        self.idx_to_class = {i: self.val_img_to_class[images[i]] for i in range(len(images))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}
        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}

    def _make_dataset(self, Train=True):
        self.images = []
        self.targets = []
        if Train:
            img_root_dir = self.train_dir
            list_of_dirs = [target for target in self.class_to_tgt_idx.keys()]
        else:
            img_root_dir = self.val_dir
            list_of_dirs = ["images"]

        for tgt in list_of_dirs:
            dirs = os.path.join(img_root_dir, tgt)
            if not os.path.isdir(dirs):
                continue

            for root, _, files in sorted(os.walk(dirs)):
                for fname in sorted(files):
                    if (fname.endswith(".JPEG")):
                        path = os.path.join(root, fname)
                        if Train:
                            item = (path, self.class_to_tgt_idx[tgt])
                            self.targets.append(self.class_to_tgt_idx[tgt])
                        else:
                            item = (path, self.class_to_tgt_idx[self.val_img_to_class[fname]])
                            self.targets.append(self.class_to_tgt_idx[self.val_img_to_class[fname]])
                        self.images.append(item)

    def return_label(self, idx):
        return [self.class_to_label[self.tgt_idx_to_class[i.item()]] for i in idx]

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, idx):
        img_path, tgt = self.images[idx]
        with open(img_path, 'rb') as f:
            sample = Image.open(img_path)
            sample = sample.convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, tgt
