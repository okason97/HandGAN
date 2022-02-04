from pycocotools.coco import COCO
from torch import zeros, tensor
from torch.utils.data import Dataset, DataLoader
import skimage.io as io
import math
import numpy as np
from skimage.draw import disk

class CocoDataset(Dataset):
    def __init__(self, root, annFile, dataType, transforms, keypoint_pad=3, shape = 64):
        self.root = root
        self.dataType = dataType
        self.annFile = annFile
        self.transforms = transforms
        self.coco = COCO(annFile)
        self.catIds = self.coco.getCatIds(catNms=['person'])
        self.ids = sorted(self.coco.getImgIds(catIds=self.catIds))
        usefulIds = []
        for id in self.ids:
            annIds = self.coco.getAnnIds(imgIds=id, catIds=self.catIds, areaRng=[8000, math.inf], iscrowd=False)
            anns = self.coco.loadAnns(annIds)
            # if len(anns)>0 and 0 not in anns[0]['keypoints']:
            if len(anns)>0 and 0 not in anns[0]['keypoints'][10:22]:
                usefulIds.append(id)
        self.ids = usefulIds
        self.keypoint_pad = keypoint_pad
        self.shape = shape

    def load_image(self, id):
        path = self.coco.loadImgs(id)[0]["file_name"]
        return io.imread('%s/images/%s/%s'%(self.root, self.dataType, path))

    def load_keypoints(self, anns, selected_person, old_shape, new_shape, shift_x, shift_y):
        keypoints = anns[selected_person]['keypoints']
        shift_x = round(shift_x/old_shape*new_shape)
        shift_y = round(shift_y/old_shape*new_shape)
        n_pose_keypoints = int(len(keypoints)/3)
        pose_map = zeros(n_pose_keypoints, new_shape, new_shape)
        for j in range(n_pose_keypoints):
            shifted_x = round(keypoints[j*3+1]/old_shape*new_shape)-shift_x
            shifted_y = round(keypoints[j*3]/old_shape*new_shape)-shift_y
            if shifted_x > 0 or shifted_y > 0:
                rr, cc = disk((shifted_x, shifted_y), self.keypoint_pad, shape=(new_shape, new_shape))
                pose_map[j, rr, cc] = 1.0

        return pose_map

    def crop_image(self, image, x, y, w, h):
        if w>h:
            pad = (w-h)/2
            dif = 0
            if 2*pad+h>image.shape[1]:
                dif = w-image.shape[1]
                pad = (image.shape[1]-h)/2
            zero_dif = min(y, int(pad+0.5))
            max_dif = min(image.shape[1]-(y+h), int(pad))
            pad_top = zero_dif+int(pad)-max_dif
            pad_bot = max_dif+int(pad+0.5)-zero_dif
            pad_left = -int(dif/2+0.5)
            pad_right = -int(dif/2)
        elif h>w:
            pad = (h-w)/2
            dif = 0
            if 2*pad+w>image.shape[0]:
                dif = h-image.shape[0]
                pad = (image.shape[0]-w)/2
            zero_dif = min(x, int(pad+0.5))
            max_dif = min(image.shape[0]-(x+w), int(pad))
            pad_top = -int(dif/2+0.5)
            pad_bot = -int(dif/2)
            pad_left = zero_dif+int(pad)-max_dif
            pad_right = max_dif+int(pad+0.5)-zero_dif
        else:
            pad_top = 0
            pad_bot = 0
            pad_left = 0
            pad_right = 0
            cropped_image = image[x:x+w, y:y+h]
        cropped_image = image[x-pad_left:x+w+pad_right, y-pad_top:y+h+pad_bot]
        return cropped_image, x-pad_left, y-pad_top

    def __getitem__(self, idx):
        id = self.ids[idx]
        image = self.load_image(id)

        # fix BW images
        if len(image.shape)<3:
            image = np.moveaxis(np.repeat(np.expand_dims(image,0),3,axis=0),0,2)

        annIds = self.coco.getAnnIds(imgIds=id, catIds=self.catIds, areaRng=[8000, math.inf], iscrowd=False)
        anns = self.coco.loadAnns(annIds)
        selected_person = 0
        # selected_person = np.random.randint(0, len(anns))

        # get annotation data for cropping
        y = round(anns[selected_person]['bbox'][0])
        x = round(anns[selected_person]['bbox'][1])
        h = round(anns[selected_person]['bbox'][2])
        w = round(anns[selected_person]['bbox'][3])

        cropped_image, shift_x, shift_y = self.crop_image(image, x, y, w, h)
        image = tensor(np.moveaxis(cropped_image, -1, 0))/255
        keypoints = self.load_keypoints(anns, selected_person, image.shape[1], self.shape, shift_x, shift_y)

        if self.transforms is not None:
            image, keypoints = self.transforms(image), keypoints

        return image, keypoints

    def __len__(self):
        return len(self.ids)

def load_images_and_poses(batch_size, transforms=None, root='./COCOPersons', shape = 64, keypoint_pad = 3):

    dataType='train2017'
    annFile = '{}/annotations/person_keypoints_{}.json'.format(root, dataType)    
    train_dataset = CocoDataset(root, annFile, dataType, transforms, shape = shape, keypoint_pad = keypoint_pad)

    dataType='val2017'
    annFile = '{}/annotations/person_keypoints_{}.json'.format(root, dataType)    
    val_dataset = CocoDataset(root, annFile, dataType, transforms, shape = shape, keypoint_pad = keypoint_pad)

    dataloaders = {'train': DataLoader(train_dataset, shuffle=True, batch_size=batch_size, pin_memory=True, num_workers=4),
                   'val': DataLoader(val_dataset, shuffle=True, batch_size=batch_size, pin_memory=True, num_workers=4)}
    image_datasets = {'train': train_dataset,
                      'val': val_dataset}
    return image_datasets, dataloaders