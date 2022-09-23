from pycocotools.coco import COCO
from PIL import Image
import os
import tqdm
import cv2
import imgviz
import numpy as np

def save_colored_mask(save_path, mask):
    lbl_pil = Image.fromarray(mask.astype(np.uint8), mode='P')
    colormap = imgviz.label_colormap(80)
    lbl_pil.putpalette(colormap.flatten())
    lbl_pil.save(save_path)


coco_root = 'D:\\Dataset\\coco\\images_mask_gray\\val2014'
annotation_file = 'D:\\Dataset\\coco\\annotations\\instances_train2014.json'

save_iscrowd = True

coco = COCO(annotation_file)
catIds = coco.getCatIds()       
imgIds = coco.getImgIds()      
print("catIds len: {}, imgIds len: {}".format(len(catIds), len(imgIds)))

cats = coco.loadCats(catIds)   
names = [cat['name'] for cat in cats]  
print(names)

img_cnt = 0
crowd_cnt = 0

for idx, imgId in tqdm.tqdm(enumerate(imgIds), ncols=100):
    if save_iscrowd:
        annIds = coco.getAnnIds(imgIds=imgId)      
    else:
        annIds = coco.getAnnIds(imgIds=imgId, iscrowd=False)  
    if len(annIds) > 0:
        image = coco.loadImgs([imgId])[0]
        ## ['coco_url', 'flickr_url', 'date_captured', 'license', 'width', 'height', 'file_name', 'id']

        h, w = image['height'], image['width']
        gt_name = image['file_name'].replace('.jpg', '.png')
        gt = np.zeros((h, w), dtype=np.uint8)
        anns = coco.loadAnns(annIds)    

        has_crowd_flag = 0
        save_flag = 0
        for ann_idx, ann in enumerate(anns):
            cat = coco.loadCats([ann['category_id']])[0]
            cat = cat['name']
            cat = names.index(cat) + 1   # re-map

            if not ann['iscrowd']:  # iscrowd==0
                segs = ann['segmentation']
                for seg in segs:
                    seg = np.array(seg).reshape(-1, 2)     # [n_points, 2]
                    cv2.fillPoly(gt, seg.astype(np.int32)[np.newaxis, :, :], int(cat))
            elif save_iscrowd:
                has_crowd_flag = 1
                rle = ann['segmentation']['counts']
                assert sum(rle) == ann['segmentation']['size'][0] * ann['segmentation']['size'][1]
                mask = coco.annToMask(ann)
                unique_label = list(np.unique(mask))
                assert len(unique_label) == 2 and 1 in unique_label and 0 in unique_label
                gt = gt * (1 - mask) + mask * 255
        save_path = os.path.join(coco_root, gt_name)
        cv2.imwrite(save_path, gt)
        img_cnt += 1
        if has_crowd_flag:
            crowd_cnt += 1

        if idx % 100 == 0:
            print('Processed {}/{} images.'.format(idx, len(imgIds)))

print('crowd/all = {}/{}'.format(crowd_cnt, img_cnt))

