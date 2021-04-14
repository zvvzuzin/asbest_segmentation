from datetime import datetime
from lxml import etree
import numpy as np
import cv2
import torch

class AverageMeter(object):
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg
    

dist_norm = lambda x : np.exp(- x ** 2 / 2) / (2 * np.pi) ** (1/2) 

def pz_approx(a, h, kernel, nums):
    t = np.linspace(min(a), max(a), nums)
    f = np.zeros(t.shape)
    for i in range(len(t)):
        for j in range(len(a)):
            f[i] += 1 / (len(a) * h) * np.sum(kernel((t[i] - a[j]) / h))
    return f

def get_quartinles(x, a):
    y = np.cumsum(a) / np.sum(a)
    indexes = np.argsort(abs(y - 0.05))[:2]
    q5 = x[indexes][0]
    indexes = np.argsort(abs(y - 0.5))[:2]
    q50 = x[indexes][0]
    indexes = np.argsort(abs(y - 0.95))[:2]
    q95 = x[indexes][0]
    return q5, q50, q95

def get_h(dist, h=0.01):
    direction = 1e-4
    a = np.array(dist)
    f = -np.Inf
    flag = True
    while abs(direction) >= 1e-5: 
        l = []
        h += direction
        if h <= 0:
            print('!')
            break 
        for i in range(len(a)):
            l.append(np.log(np.sum([dist_norm((a[i] - a[j]) / h) for j in range(len(a)) if i != j]) / (len(a) - 1) / h))
        r = np.sum(l) / len(a)              
        if r < f:
            h -= direction
            if flag:
                direction = -direction
                flag = False
            else:
                direction = direction * 1e-1
        else:
            f = r
    return h


def get_pz_quantils(dist, h, num_points = 200):
    a = np.array(dist)
    t = np.linspace(np.min(a), np.max(a), num_points)
    f = pz_approx(a, h, dist_norm, num_points)
    q5, q50, q95 = get_quartiles(t, f)
    return q5, q50, q95


def plot_dist_pz(dist, h, num_points = 200):
    a = np.array(dist)
    hist, x, *_ = np.histogram(a, bins=10);
    hist = hist / sum(hist) / (x[1] - x[0])
    plt.bar([(x[i] + x[i+1]) / 2 for i in range(len(x) - 1)], hist, width=(x[1] - x[0]))
    t = np.linspace(np.min(a), np.max(a), num_points)
    f = pz_approx(a, h, dist_norm, num_points)
    q5, q50, q95 = get_quartiles(t, f)
    plt.plot(t, f, 'r')
    plt.xlabel('Удельное содержание асбеста')
    plt.vlines(np.array([q5, q50, q95]), 0, max(hist))
    print('Квантиль на уровне доверительной вероятности 0.05 = ', q5)
    print('Квантиль на уровне доверительной вероятности 0.50 = ', q50)
    print('Квантиль на уровне доверительной вероятности 0.95 = ', q95)
    return q5, q50, q95


def parse_anno_file(cvat_xml):
    root = etree.parse(cvat_xml).getroot()
    anno = []
    for image_tag in root.iter('image'):
        image = {}
        for key, value in image_tag.items():
            image[key] = value
        image['shapes'] = []
        for poly_tag in image_tag.iter('polygon'):
            polygon = {'type': 'polygon'}
            for key, value in poly_tag.items():
                polygon[key] = value
            image['shapes'].append(polygon)

        image['shapes'].sort(key=lambda x: int(x.get('z_order', 0)))
        anno.append(image)

    return anno

def create_mask_file(annotation, label, binary=True):
    size = (int(annotation['height']), int(annotation['width']))
#     labels = set([ob['label'] for ob in annotation['shapes']])
    mask = np.zeros(size, dtype=np.uint8)
    color = 1
    for shape in annotation['shapes']:
        if label == shape['label']:
            points = [tuple(map(float, p.split(','))) for p in shape['points'].split(';')]
            points = np.array([(int(p[0]), int(p[1])) for p in points])
            if binary:
                mask = cv2.fillPoly(mask, [points], color=255)
            else:
                mask = cv2.fillPoly(mask, [points], color=color)
                color += 1
    if binary:
        mask = mask > 0 
    return mask
        

def get_time(file):
    return datetime.strptime(file.split('_')[1] + '_' + file.split('_')[2], '%H:%M:%S_%d-%m-%Y')


def get_clip_limit(file):
    date = get_time(file)
    sample = file.split('_')[0]
    if date.day == 16:
        clip_limit = 1.1
    elif date.day == 5 and int(sample) <= 4:
        clip_limit = 1.1
    elif date.day == 5 and int(sample) >= 12:
        clip_limit = 2.0
    elif date.day == 5 and int(sample) >= 5 and int(sample) <= 11:
        clip_limit = 2.5
    return clip_limit


def imp_cont_img_file(file, clip_limit=None):
    if clip_limit is None:
        clip_limit = get_clip_limit(file.split('/')[-1])
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8,8))
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE).astype(float) / 255
    img -= 0.1
    img = (np.clip(img / img.max(), 0, 1) * 255).astype(np.uint8)
    img = clahe.apply(img).astype(float) / 255
    return img


def preprocess_image(image, clip_limit=None, tile=(8,8)):
    if clip_limit:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile)
    else:
        clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=tile)
    if image.dtype == np.uint8:
        image = image.astype(float) / 255
    image -= 0.1
    image = (np.clip(image / image.max(), 0, 1) * 255).astype(np.uint8)
    image = clahe.apply(image).astype(float) / 255
    return image


def big_image_predict(model, image, crop_size, inp_size, normalize=True, device='cpu'):
    
    h, w = image.shape[:2]
    st_mask = np.zeros(image.shape[:2], dtype = float)
    asb_mask = np.zeros(image.shape[:2], dtype = float)
    mean_mask = np.zeros(image.shape[:2], dtype = float)
    num_img_y = int(np.ceil(h / crop_size[0])) * 2 - 1
    num_img_x = int(np.ceil(w / crop_size[1])) * 2 - 1
    image = preprocess_image(image, 1.2)
    
    
    if normalize:
        image = (image - 0.5) / 0.5 
    
    for j in range(num_img_y):
        for i in range(num_img_x):
            part_image = image[int(j / 2  * crop_size[0]):int((j / 2 + 1) * crop_size[0]), 
                               int(i / 2 * crop_size[1]):int((i / 2 + 1) * crop_size[1])].copy()
            mean_mask[int(j / 2  * crop_size[0]):int((j / 2 + 1) * crop_size[0]), 
                      int(i / 2 * crop_size[1]):int((i / 2 + 1) * crop_size[1])] += 1
            init_shape = part_image.shape[:2]
            
            if part_image.shape[0] < crop_size[0]:
                part_image = np.concatenate((part_image, 
                                             np.zeros((crop_size[1] - part_image.shape[0], part_image.shape[1]))), axis=0)
            
            if part_image.shape[1] < crop_size[1]:
                part_image = np.concatenate((part_image, 
                                             np.zeros((part_image.shape[0], crop_size[0] - part_image.shape[1]))), axis=1)
            
            part_image = cv2.resize(part_image, inp_size, interpolation = 1)
            
            part_image = torch.tensor(np.expand_dims(np.expand_dims(part_image, axis=0), axis=0)).to(device).float()
            
            model.eval()
            out_mask = model(part_image).cpu()
            out_mask = np.squeeze(out_mask.cpu().detach().numpy())
            out_st_mask = out_mask[0]
            out_asb_mask = out_mask[1]
            out_st_mask = cv2.resize(out_st_mask, (crop_size[1], crop_size[0]), interpolation=0) [:init_shape[0], :init_shape[1]]
            out_asb_mask = cv2.resize(out_asb_mask, (crop_size[1], crop_size[0]), interpolation=0) [:init_shape[0], :init_shape[1]]
            
            st_mask[int(j / 2  * crop_size[0]):int((j / 2 + 1) * crop_size[0]), 
                    int(i / 2 * crop_size[1]):int((i / 2 + 1) * crop_size[1])] += out_st_mask
            asb_mask[int(j / 2  * crop_size[0]):int((j / 2 + 1) * crop_size[0]), 
                     int(i / 2 * crop_size[1]):int((i / 2 + 1) * crop_size[1])] += out_asb_mask
    
    if normalize:
        image = (image + 1) / 2
    
    return np.clip(image, 0, 1), np.clip(st_mask / mean_mask, 0, 1), np.clip(asb_mask / mean_mask, 0, 1)

true_results = {
    5 : {
        1 : 1.48,
        2 : 2.74, 
        3 : 2.86,
        4 : 2.52,
        5 : 3.16,
        6 : 1.36,
        7 : 2.98,
        8 : 1.79,
        9 : 2.81,
        11 : 2.57,
        12 : 2.80,
        13 : 2.40,
        14 : 2.44,
        18 : 2.29,
        19 : 2.64,
        20 : 3.28,
    },
    16 : {
        1 : 1.94,
        2 : 1.89,
        3 : 2.89,
        4 : 2.62,
        5 : 1.12,
        6 : 3.86,
        7 : 2.72,
        8 : 2.84,
        9 : 2.37,
        10 : 0.29,
        11 : 2.14,
        12 : 3.17,
        13 : 3.04,
        14 : 2.94,
        15 : 2.40,
        16 : 2.35,
        17 : 3.09,
        18 : 2.80,
        19 : 2.38,
        20 : 0.22,
        21 : 1.01,
        22 : 4.36,
        23 : 2.64,
        24 : 1.83,
        25 : 2.50,
        26 : 2.44,
        27 : 2.31,
        28 : 2.68,
        29 : 2.96,
        30 : 0.15,
    }
}