import os

import torch
import scipy.io
from onegan.io.loader import BaseDataset
from onegan.io.functional import load_image


from lib.lsun_room_api.lsun_room.edge import gen_edge_map, gen_corner_map

class SegmentationPair():

    def __init__(self,
                 target_size=None,
                 final_transform=False,
                 random_flip=False, random_crop=False, color_jiiter=False):
        self.target_size = target_size
        self.final_transform = final_transform
        self.random_flip = random_flip
        self.random_crop = random_crop
        self.color_jiiter = color_jiiter

    def __call__(self, image, segmentation, random=False) -> tuple:
        image = image.convert('RGB')
        segment = segmentation.convert('L')

        image, segment = self.tf_random_flip(image, segment)
        image, segment = self.tf_random_crop(image, segment)

        image = F.resize(image, self.target_size, interpolation=Image.BILINEAR)
        segment = F.resize(segment, self.target_size, interpolation=Image.NEAREST)

        return self._transform(image, segment)

    def _transform(self, image, segmentation) -> tuple:
        if self.final_transform:
            image = T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])(F.to_tensor(image))
            segmentation = torch.from_numpy(np.array(segmentation) - 1).long()  # make 0 into 255 as ignore index
        return image, segmentation

    def tf_random_flip(self, image, segment):
        if self.random_flip and random.random() >= 0.5:
            image = F.hflip(image)
            segment = F.hflip(segment)
        return image, segment

    def tf_random_crop(self, image, segment):
        if self.random_crop:
            i, j, h, w = F.RandomResizedCrop.get_params(image)
            image = F.crop(image, i, j, h, w)
            segment = F.crop(segment, i, j, h, w)
        return image, segment

def get_meta(dataset_root, phase):
    phase = {'train': 'training', 'val': 'validation', 'test': 'testing'}[phase]
    #print('-------------------------------', phase, dataset_root)
    mat = scipy.io.loadmat(os.path.join(dataset_root, f'{phase}.mat'))[phase][0]
    return [dict(name=m[0][0], scene=m[1][0], type=m[2][0][0], points=m[3], resolution=m[4][0]) for m in mat]


class LsunRoomDataset(BaseDataset):

    num_classes = 5
    image_folder = 'images/'
    layout_folder = 'layout_seg_images/'

    def __init__(self, phase, root, target_size):
        self.root = root
        self.target_size = target_size
        self.meta = get_meta(dataset_root=root, phase=phase)
        self.paired_transform = SegmentationPair(target_size, final_transform=True)

    def __getitem__(self, index):
        e = self.meta[index]
        image_path = os.path.join(self.root, self.image_folder, f'{e["name"]}.jpg')
        layout_path = os.path.join(self.root, self.layout_folder, f'{e["name"]}.png')

        image, layout = self.paired_transform(load_image(image_path), load_image(layout_path))
        edge = torch.from_numpy(self.load_edge_map(index) / 255).float()
        return {'image': image, 'label': layout, 'edge': edge, 'type': e['type']}

    def load_edge_map(self, index):
        e = self.meta[index]
        return gen_edge_map(e, self.target_size, self.edge_width, self.edge_sigma)

    def load_corner_map(self, index):
        e = self.meta[index]
        return gen_corner_map(e, self.target_size).astype('float32')

    def __len__(self):
        return len(self.meta)

    @property
    def edge_width(self):
        if not hasattr(self, '_edge_width'):
            self._edge_width = 30
            self.edge_sigma = 7
        return self._edge_width

    @edge_width.setter
    def edge_width(self, width):
        self._edge_width = int(width) if width > 4 else 4
        self.edge_sigma = 7 if width > 4 else 2
