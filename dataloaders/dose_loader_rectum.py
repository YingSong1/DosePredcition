import os, pdb
import pydicom
import SimpleITK as sitk
import numpy as np
import operator
from PIL import Image, ImageOps
import warnings
import cv2
import collections
from torch.utils.data import Dataset
from tqdm import tqdm

def slice_order(sample_path):
    """
    Takes path of directory that has the DICOM images and returns
    a ordered list that has ordered filenames
    Inputs
        path: path that has .dcm images
    Returns
        ordered_slices: ordered tuples of filename and z-position
    """
    slices = []
    for filename in os.listdir(sample_path):
        if filename.startswith("CT"):
            f = pydicom.read_file(os.path.join(sample_path, filename))
            slices.append(f)
            
    slice_dict = {s.SOPInstanceUID: s.ImagePositionPatient[-1] for s in slices}
    ordered_slices = sorted(slice_dict.items(), key=operator.itemgetter(1), reverse=True)
    return [slice_info[0] for slice_info in ordered_slices]

class Dose_online_avoid(Dataset):
    def __init__(self, data_root, 
                        sample_txt,
                        mask_dict,
                        transforms,
                        CT_prefixs = ["CT.", "CT"],
                        dose_prefix = "RD",
                        required_masks = [1,2,3,4,5]
                        ):
        """this class use the ptv to avoid the OARs, and use OARs and ptv to avoid the body"""
        Record = collections.namedtuple('Record', ['sample_name', 'sample_path',  'image_raw_name', 'image_name','source_masks', 
                                                   'ct_lefttop_coords', 'ct_rightdown_coords', 'rd_slice', 'dose_max', 'dose_min'])
        self.mask_dict = mask_dict
        self.transforms = transforms
        self.dose_prefix = dose_prefix
        self.records = []
        sample_masks = self._resolve_samplemask_txt(sample_txt)
        
        for (sample_name, source_masks) in tqdm(sample_masks.items()):
            dose_max, dose_min = -1, 1e8
            sample_path = os.path.join(data_root, sample_name)
            ct_raw_names = slice_order(sample_path)
            rd_arr, rd_list = self._get_dose(sample_path)
            dose_max, dose_min = np.max(rd_arr), np.min(rd_arr)
            rd_lefttop_loc, rd_rightdown_loc = self.get_rd_locs(rd_list[0])
            for raw_name in ct_raw_names:
                for CT_prefix in CT_prefixs:
                    img_name = "{}{}.dcm".format(CT_prefix, raw_name)
                    if os.path.exists(os.path.join(sample_path, img_name)): break
                
                ct_obj = pydicom.read_file(os.path.join(sample_path, img_name))
                ct_lefttop_coords, ct_rightdown_coords = self.get_ct_coord(ct_obj, rd_lefttop_loc, rd_rightdown_loc)
                rd_slice = self.get_rd_byz(rd_arr, 
                                           rd_start_z = float(rd_list[0].ImagePositionPatient[-1]), 
                                           rd_z_offsets = rd_list[0].GridFrameOffsetVector, 
                                           target_z = float(ct_obj.ImagePositionPatient[-1]))
                if rd_slice is not None: # only when the ct slice has corresponding rd_slice
                    record = Record(sample_name, sample_path, raw_name, img_name, source_masks, ct_lefttop_coords, ct_rightdown_coords, rd_slice, dose_max, dose_min)
                    if self._check_mask(record, required_masks): # only when the ct slice has corresponding mask
                        self.records.append(record)
    
    def _resolve_samplemask_txt(self, sample_txt):
        fp = open(sample_txt, "r")
        lines = fp.readlines()
        fp.close()
        sample_masks = {}
        for line in lines:
            line = line.strip()
            if line != "" and not line.startswith("#"):
                sample_name = line.split("\t")[0]
                mask_names = line.split("\t")[1:]
                sample_masks[sample_name] = mask_names
        return sample_masks
    
    def _get_dose(self, sample_path):
        rd_total = []
        rd_fraction_group = []
        for filename in os.listdir(sample_path):
            filepath = os.path.join(sample_path, filename)
            if filename.startswith('RD'):
                rd_obj = pydicom.read_file(filepath)
                if "ReferencedFractionGroupSequence" in rd_obj.ReferencedRTPlanSequence[0].dir():
                    rd_fraction_group.append(rd_obj)
                else:
                    rd_total.append(rd_obj)
        ReferencedSOPInstanceUID = set([rd_obj.ReferencedRTPlanSequence[0].ReferencedSOPInstanceUID for rd_obj in rd_fraction_group+rd_total])
        if len(ReferencedSOPInstanceUID)!=1:
            raise(Exception("ReferencedSOPInstanceUIDs are different, please check the folder:%s"%sample_path))
        if len(rd_total)==1:
            rd_obj = rd_total[0]
            dose_arr = rd_obj.pixel_array * float(rd_obj.DoseGridScaling)
            return dose_arr, rd_total
        if len(rd_total)>1:
            raise(Exception("number of rd total file over 1, please check the folder:%s"%sample_path))
        if rd_fraction_group:
            dose_arr = np.sum(np.stack([rd_obj.pixel_array * float(rd_obj.DoseGridScaling) for rd_obj in rd_fraction_group]), axis=0)
            return dose_arr, rd_fraction_group
        else:
            raise(Exception(sample_path+"has no correspondding RD files"))
    
    def get_rd_locs(self, rd_obj):
        """get the left top and right down locations"""
        lefttop_loc = [float(v) for v in rd_obj.ImagePositionPatient]
        col, row = rd_obj.Columns, rd_obj.Rows
        col_spacing, row_spacing = float(rd_obj.PixelSpacing[0]), float(rd_obj.PixelSpacing[1])
        rightdown_loc = [lefttop_loc[0] + col_spacing*col, 
                           lefttop_loc[1] + row_spacing*row, 
                           lefttop_loc[-1]]
        return lefttop_loc, rightdown_loc

    def get_ct_coord(self, ct_obj, lefttop_loc, rightdown_loc, bias = 1):
        """get the pixel location of ct's data array according to the left top and right down coordinates of rd object"""
        ct_lefttop_loc = [float(v) for v in ct_obj.ImagePositionPatient]
        col_spacing, row_spacing = float(ct_obj.PixelSpacing[0]), float(ct_obj.PixelSpacing[1])
        col, row = ct_obj.Columns, ct_obj.Rows
        start_col, end_col, start_row, end_row = 0, 0, 0, 0 
        
        for col_idx in range(col):
            col_coord = ct_lefttop_loc[0] + col_idx*col_spacing
            if abs(col_coord - lefttop_loc[0]) < bias:
                start_col = col_idx

            if abs(col_coord - rightdown_loc[0]) < bias:
                end_col = col_idx

            if start_col != 0 and end_col != 0:
                break
        for row_idx in range(row):
            row_coord = ct_lefttop_loc[1] + row_idx*row_spacing
            if abs(row_coord - lefttop_loc[1]) < bias:
                start_row = row_idx

            if abs(row_coord - rightdown_loc[1]) < bias:
                end_row = row_idx

            if start_row != 0 and end_row != 0:
                break
        # this means the end coordinate of the rd cann't be reached by the ct slice
        if end_row == 0:
            end_row = 511
        if end_col == 0:
            end_col = 511
        
        return [start_col, start_row, ct_lefttop_loc[-1]], [end_col+1, end_row+1, ct_lefttop_loc[-1]]
    
    def get_rd_byz(self, rd_arr, rd_start_z, rd_z_offsets, target_z, bias=1.5):
        """get the slice of rd according to the z coordinate"""
        rd_idx = -1
        for i in range(len(rd_z_offsets)):
            z_loc = rd_start_z + float(rd_z_offsets[i])
            if abs(z_loc - target_z) <= bias:
                return rd_arr[i]
        return None
    
    def _check_mask(self, record, required_masks):
        """check whether the CT slice record has corresponding mask in the source_masks and the record has the required_masks"""
        mask_exist, has_requireed_mask = False, False
        for source_mask in record.source_masks:
            mask_path = os.path.join(record.sample_path, "mask", source_mask)
            target_mask_path = os.path.join(mask_path, record.image_raw_name + ".bmp")
            if os.path.exists(target_mask_path):
                for (mask_idx, target_masks) in self.mask_dict.items():
                    if source_mask in target_masks and mask_idx in required_masks:
                        return True
        return False
    
    def _resolve_record(self, record):
        img_path = os.path.join(record.sample_path, record.image_name)
        img = sitk.GetArrayFromImage(sitk.ReadImage(img_path))[0]
        img = img[record.ct_lefttop_coords[1]: record.ct_rightdown_coords[1], 
                  record.ct_lefttop_coords[0]: record.ct_rightdown_coords[0]]
        mask_list, mask_idxs = [], []
        for (mask_idx, target_masks) in self.mask_dict.items():
            for source_mask in record.source_masks:
                if source_mask in target_masks:
                    mask = self._get_mask(record.sample_path, record.image_raw_name, source_mask)
                    if mask is None: # the mask may be None
                        mask = np.zeros((record.ct_rightdown_coords[1]-record.ct_lefttop_coords[1],
                                        record.ct_rightdown_coords[0]-record.ct_lefttop_coords[0]),dtype=np.uint8)
                    else:
                        mask = mask[record.ct_lefttop_coords[1]: record.ct_rightdown_coords[1], 
                                      record.ct_lefttop_coords[0]: record.ct_rightdown_coords[0]]
                    mask_list.append(mask)
                    mask_idxs.append(mask_idx)
        # sort the order of masks according to its label
        sorted_masks = [mask for (idx, mask) in sorted(zip(mask_idxs, mask_list))]
        return img, sorted_masks, record.rd_slice
        
    def _get_mask(self, sample_path, ct_raw_name, mask_name):
        
        mask_path = os.path.join(sample_path, "mask", mask_name)
        if not os.path.exists(mask_path): 
            return None
        
        target_mask_path = os.path.join(mask_path, ct_raw_name + ".bmp")
        if os.path.exists(target_mask_path):
            mask = cv2.imread(target_mask_path)[:, :, 0]
            np.place(mask, mask == 255, 1)
            return mask
        else:
            return None
    
    def _avoid_ptv(self, masks, ptv_idx, other_mask_idxs):
        """avoid the ptv in other source mask"""
        ptv = masks[ptv_idx]
        for mask_idx in other_mask_idxs:
            target_mask = masks[mask_idx]
            inter = (ptv & target_mask)
            np.place(target_mask, inter, 0)
            masks[mask_idx] = target_mask
        return masks
    
    def _avoid_body(self, masks, body_idx, other_mask_idxs):
        """avoid other mask in the body"""
        body = masks[body_idx]
        for mask_idx in other_mask_idxs:
            target_mask = masks[mask_idx]
            np.place(body, target_mask!=0, 0)
        masks[body_idx] = body
        return masks
    
    def __len__(self):
        return len(self.records)
    
    def __getitem__(self, idx):
        img, masks, rd_slice = self._resolve_record(self.records[idx])
        # use ptv to avoid OARs, note that the index of ROI is smaller than the mask_dict
        masks = self._avoid_ptv(masks, ptv_idx=0, other_mask_idxs=[1, 2, 3, 4])
        # use ptv and OARs to avoid the body
        masks = self._avoid_body(masks, body_idx = 5, other_mask_idxs = [0, 1, 2, 3, 4])
        
        sample = {'image': img, 'masks': masks, 'rd_slice': rd_slice, 'sample_name': self.records[idx].sample_name, 'image_name':self.records[idx].image_name, 'dose_max':self.records[idx].dose_max, 'dose_min':self.records[idx].dose_min}
        
        if self.transforms is not None:
            sample = self.transforms(sample)
        
        return sample