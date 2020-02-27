import torch
import math
import numbers
import random, pdb
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

class FilterHU(object):
    def __init__(self, HU_min, HU_max):
        self.HU_min = HU_min
        self.HU_max = HU_max
    
    def __call__(self, sample):
        image = sample["image"].astype(np.int32)
        
        np.place(image, image > self.HU_max, self.HU_max)
        np.place(image, image < self.HU_min, self.HU_min)
        
        sample["image"] = image
        return sample

class Arr2image(object):
    def __call__(self, sample):
        image = sample["image"]
        rd_slice = sample["rd_slice"]
        masks = sample["masks"]
        if type(image) == np.ndarray:
            image = Image.fromarray(image)
        
        if type(rd_slice) == np.ndarray:
            rd_slice = Image.fromarray(rd_slice)
        
        target_masks = []
        for mask in masks:
            if type(mask) == np.ndarray:
                mask = Image.fromarray(mask)
            target_masks.append(mask)
        
        sample["image"] = image
        sample["masks"] = target_masks
        sample["rd_slice"] = rd_slice
        return sample

class AlignDose(object):
    def __call__(self, sample):
        """upsample dose slice to match the CT's resolution"""
        image = sample["image"]
        rd_slice = sample["rd_slice"]
        
        w, h = image.size
        rd_slice = rd_slice.resize((w, h), Image.NEAREST)
        
        sample["rd_slice"] = rd_slice
        return sample

    
class AlignCT(object):
    def __call__(self, sample):
        """downsample CT and OARs to match the dose's resolution"""
        image = sample["image"]
        masks = sample["masks"]
        rd_slice = sample["rd_slice"]
        
        w, h = rd_slice.size
        
        image = image.resize((w, h), Image.BILINEAR)
        masks = [mask.resize((w, h), Image.NEAREST) for mask in masks]
        sample["image"] = image
        sample["masks"] = masks
        return sample

    
class Padding(object):
    """padding zero to image to match the maximum value of target width or height"""
    def __init__(self, size, fill=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.fill = fill
    
    def __call__(self, sample):
        image = sample['image']
        masks = sample['masks']
        rd_slice = sample['rd_slice']
        
        w, h = image.size
        target_h, target_w = self.size
        
        left = top = right = bottom = 0
        doit = False
        if target_w > w:
            delta = target_w - w
            left = delta // 2
            right = delta - left
            doit = True
            
        if target_h > h:
            delta = target_h - h
            top = delta // 2
            bottom = delta - top
            doit = True
        if doit:
            image = ImageOps.expand(image, border=(left, top, right, bottom), fill=self.fill)
            target_masks = [ImageOps.expand(mask, border=(left, top, right, bottom), fill=self.fill) for mask in masks]
            masks = target_masks
            rd_slice = ImageOps.expand(rd_slice, border=(left, top, right, bottom), fill=self.fill)
            
        sample["image"] = image
        sample["masks"] = masks
        sample["rd_slice"] = rd_slice
        return sample

class RandomHorizontalFlip(object):
    def __call__(self, sample):
        image = sample["image"]
        masks = sample["masks"]
        rd_slice = sample["rd_slice"]
        
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            masks = [mask.transpose(Image.FLIP_LEFT_RIGHT) for mask in masks]
            rd_slice = rd_slice.transpose(Image.FLIP_LEFT_RIGHT)

        sample["image"] = image
        sample["masks"] = masks
        sample["rd_slice"] = rd_slice
        return sample


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        image = sample["image"]
        masks = sample["masks"]
        rd_slice = sample["rd_slice"]

        rotate_degree = random.random() * 2 * self.degree - self.degree
        image = image.rotate(rotate_degree, Image.BILINEAR)
        masks = [mask.rotate(rotate_degree, Image.NEAREST) for mask in masks]
        rd_slice = rd_slice.rotate(rotate_degree, Image.NEAREST)

        sample["image"] = image
        sample["masks"] = masks
        sample["rd_slice"] = rd_slice
        return sample    
    
class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, sample):
        image = sample['image']
        masks = sample['masks']
        rd_slice = sample['rd_slice']
        
        w, h = image.size
        target_h, target_w = self.size

        w_start = int(round((w - target_w) / 2.))
        h_start = int(round((h - target_h) / 2.))

        image = image.crop((w_start, h_start, w_start + target_w, h_start + target_h))
        masks = [mask.crop((w_start, h_start, w_start + target_w, h_start + target_h)) for mask in masks]
        rd_slice = rd_slice.crop((w_start, h_start, w_start + target_w, h_start + target_h))

        sample["image"] = image
        sample["masks"] = masks
        sample["rd_slice"] = rd_slice
        return sample    
    
class NormalizeCT(object):
    def __init__(self, HU_min, HU_max):
        self.HU_min = HU_min
        self.HU_max = HU_max
    
    def __call__(self, sample):
        image = sample["image"]
        image = (np.array(image) - float(self.HU_min)) / (self.HU_max - self.HU_min)
        
        sample["image"] = image
        return sample    

class NormalizeDosePerSample(object):
    def __call__(self, sample):
        """normalize the dose per sample using the max and min values of the dose in the sample"""
        rd_slice = np.array(sample["rd_slice"]).astype(np.float32)
        dose_max, dose_min = sample["dose_max"], sample["dose_min"]
        if dose_max > 0:
            rd_slice = (rd_slice - dose_min) / (dose_max - dose_min)
        sample["rd_slice"] = rd_slice
        return sample
        
    
class Stack2Tensor(object):
    def __call__(self, sample):
        image = np.expand_dims(np.array(sample["image"]), 0)
        masks = sample["masks"]
        w, h = masks[0].size
        target_masks = np.zeros((len(masks), h, w))
        for i, mask in enumerate(masks):
            target_masks[i] = np.array(mask)
        
        rd_slice = np.array(sample["rd_slice"])
        
        stacked = np.concatenate([image, target_masks])
        stacked = torch.from_numpy(stacked).float()
        rd_slice = torch.from_numpy(rd_slice).float()
        
        sample["image"] = image
        sample["masks"] = target_masks
        
        sample["input"] = stacked
        sample["rd_slice"] = rd_slice
        return sample
