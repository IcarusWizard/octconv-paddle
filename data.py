import os
import cv2
import paddle
from paddle.io import Dataset


def find_classes(directory):
    """Finds the class folders in a dataset."""
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx

def make_samples(directory, classes):
    samples = []
    for idx, cls in enumerate(classes):
        cls_dir = os.path.join(directory, cls)
        for image_file in os.listdir(cls_dir):
            samples.append((os.path.join(cls_dir, image_file), idx))
    # samples = sorted(samples, key=lambda x: os.path.split(x[0])[1])
    return samples

class ImageNetDataset(Dataset):
    def __init__(self, root, split='train', transforms=None):
        super().__init__()
        self.root = root
        self.split = split
        self.image_folder = os.path.join(root, split)
        self.classes, self.class_to_idx = find_classes(self.image_folder)
        self.samples = make_samples(self.image_folder, self.classes)
        self.transforms = transforms

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]

        # NOTE: paddle use cv2 to load image. By default, the data is in BGR format.
        # So, here we convert it to RGB to be consistent with other frameworks. 
        image = paddle.dataset.image.load_image(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transforms:
            image = self.transforms(image)
        return image, label

if __name__ == '__main__':
    import paddle.vision.transforms as T
    trans = T.Compose([
        T.Resize((256, 256)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.Transpose([2, 0, 1]),
        T.Normalize([123.68, 116.779, 103.939], [58.393, 57.12, 57.375])
    ])
    dataset = ImageNetDataset('/mnt/d/imagenet', split='val', transforms=trans)
    print(len(dataset))
    print(dataset[0], dataset[0][0].shape)
    from paddle.io import DataLoader
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    print(next(iter(loader)))
    print(next(iter(loader))[0].shape)