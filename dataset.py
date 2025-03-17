
#%%
import os
from pydicom import dcmread
import torch
# from torchvision.transforms import v2 as T
from torchvision import transforms as T
import torchvision.transforms.functional as TF
from skimage import img_as_float32
from image_patcher import ImagePatcher
import logging

logger = logging.getLogger(__name__)
#%%
class BreastCancerDataset(torch.utils.data.Dataset):
    def __init__(self, root: os.PathLike, df, view: list, transforms,
                 conv_to_bag=True, bag_size=-1,
                 img_size=[3518, 2800], is_multimodal=False,
                 patch_size=128, overlap=0.5, empty_thresh=0.75):
        self.view = view
        self.root = root
        self.df = df
        self.multimodal = is_multimodal
        logger.info(f"Multimodal: {self.multimodal}")
        self.img_size = img_size
        logger.info(f"Image size: {self.img_size}")
        self.views, self.dicoms, self.class_name = self.__select_view()
        self.transforms = transforms
        self.convert_to_bag = conv_to_bag
        self.patcher = ImagePatcher(patch_size=patch_size, overlap=overlap, empty_thresh=empty_thresh, bag_size=bag_size)
        self.tiles = self.patcher.get_tiles(self.img_size[0], self.img_size[1])

    def __getitem__(self, idx):
        os.chdir(os.path.join(self.root, self.class_name[idx]))

        if self.multimodal:
            img, dcm = self.load_dcm_multimodal(idx)
            _, height, width = img.shape
        else:
            img, dcm = self.load_dcm_unimodal(idx)
            _, height, width = img.shape
        if (height != self.img_size[0]) and (width != self.img_size[1]):
                t = T.Resize((self.img_size[0], self.img_size[1]), antialias=True)
                img = t(img)
        # img = img/torch.max(img)

        target = {}
        target["label"] = torch.tensor(1. if self.class_name[idx] in ['Malignant', 'Lymph_nodes'] else 0.)
        target["class"] = self.class_name[idx]
        
        meda_data = {
            "view": self.views[idx],
            "file": self.dicoms[idx],
            "patient_id": dcm.PatientID,
            "age": self.__get_age(dcm),
            "laterality": self.__get_laterality(dcm),
            "img_h": height,
            "img_w": width
        }

        if meda_data["laterality"] == 'R':
            t = T.RandomHorizontalFlip(p=1.0)
            img = t(img)
        # translation -px (white strips near image border)
        img = TF.affine(img, angle=0, translate=(-20, 0), scale=1, shear=0)
        
        if self.convert_to_bag:
            instances, instances_ids, instances_coords = self.patcher.convert_img_to_bag(img)
            if self.transforms:
                instances = torch.stack([self.transforms(image) for image in instances]) 
            data = {'image': instances, 'target': target, 'metadata': meda_data}
            data['metadata']['tiles_indices'] = instances_ids
        else:
            data = {'image': img, 'target': target, 'metadata': meda_data}
        
        return data

    def __len__(self):
        return len(self.dicoms)

    def load_dcm_multimodal(self, idx):
        CC_path = None
        MLO_path = None
        for i in range(len(self.dicoms[idx])):
            if "CC" in self.dicoms[idx][i]:
                CC_path = self.dicoms[idx][i]
            if "ML" in self.dicoms[idx][i] or "MO" in self.dicoms[idx][i]:
                MLO_path = self.dicoms[idx][i]
        if CC_path is None or MLO_path is None:
            print(self.dicoms[idx])
            raise ValueError("CC or MLO not found")
        dcm = dcmread(CC_path)
        # img_CC = dcm.pixel_array
        # img_CC = img_CC/4095
        # img_CC = img_as_float32(img_CC)
        img_CC = self.__normalize_dicom(dcm)
        img_CC = torch.from_numpy(img_CC).unsqueeze(0).repeat(3, 1, 1)

        dcm = dcmread(MLO_path)
        # img_MLO = dcm.pixel_array
        # img_MLO = img_MLO/4095
        # img_MLO = img_as_float32(img_MLO)
        img_MLO = self.__normalize_dicom(dcm)
        img_MLO = torch.from_numpy(img_MLO).unsqueeze(0).repeat(3, 1, 1)

        img = torch.cat((img_MLO, img_CC), dim=1)
        return img, dcm

    def load_dcm_unimodal(self, idx, img_only=False):
        dcm = dcmread(self.dicoms[idx])
        img = self.__normalize_dicom(dcm)
        # img = dcm.pixel_array
        height, width = img.shape
        # img = img/4095
        # img = img_as_float32(img)
        img = torch.from_numpy(img).unsqueeze(0).repeat(3, 1, 1)
        if img_only:
            return img
        else:
            return img, dcm, height, width

    def __select_view(self):
        '''Select only given view(s) and return list of filenames
           and class names(folder names)
        '''
        class_names_list = []
        filenames_list = []
        view_list = []
        patients = self.df.to_dict('records')

        if self.multimodal:
            for patient in patients:
                # take 2 first rows (sorted) if 2 or more rows are present
                if 'LCC' in patient['view'] and 'LMLO' in patient['view']:
                    # if patient['class'][0] in ['Malignant', 'Lymph_nodes']:
                    flist = [f for f in patient['filename'] if 'L_C' in f or 'L_M' in f]
                    if len(flist) != 2:
                        print(f"Patient {patient['filename']} has invalid combination of L CC or MLO view")
                        continue
                    filenames_list.append(flist)
                    class_names_list.append(patient['class'][0])
                    view_list.append('Left')
                # take 2 last rows (sorted) if 2 or more rows are present
                elif 'RCC' in patient['view'] and 'RMLO' in patient['view']:
                    # if patient['class'][-1] in ['Malignant', 'Lymph_nodes']:
                    flist = [f for f in patient['filename'] if 'R_C' in f or 'R_M' in f]
                    if len(flist) != 2:
                        print(f"Patient {patient['filename']} has invalid combination of R CC or MLO view")
                        continue
                    filenames_list.append(flist)
                    class_names_list.append(patient['class'][-1])
                    view_list.append('Rigth')
        else:
            for patient in patients:
                for item in range(len(patient['class'])):
                    for v in self.view:
                        if patient['view'][item].__contains__(v):
                            class_names_list.append(patient['class'][item])
                            filenames_list.append(patient['filename'][item])
                            view_list.append(patient['view'][item])
            # if patient['class'][item].find('Malignant') is not -1:
            #     class_names_list.append(patient['class'][item])
            #     filenames_list.append(patient['filename'][item])
            #     view_list.append(patient['view'][item])
            # if patient['class'][item].find('Lymph_nodes') is not -1:
            #     class_names_list.append(patient['class'][item])
            #     filenames_list.append(patient['filename'][item])
            #     view_list.append(patient['view'][item])

        return view_list, filenames_list, class_names_list

    def __get_age(self, dcm):
        '''Read Patient's age from DICOM data'''
        dcm_tag = (0x0010, 0x1010)
        # 0x0010, 0x1010 - Patient's Age in form 'dddY'
        idx_end = str(dcm[dcm_tag]).find('Y')
        return int(str(dcm[dcm_tag])[idx_end-3:idx_end])

    def __get_laterality(self, dcm):
        '''
        Read Image Laterality from DICOM data
        Returns 'L' or 'R' as string type dependent on breast laterality
        '''
        return dcm.ImageLaterality

    def __normalize_dicom(self, dcm):
        """Normalize DICOM pixel values dynamically using BitsStored."""
        bits_stored = dcm.BitsStored  # Extract actual bit depth
        max_val = (2 ** bits_stored) - 1  # Compute max pixel value
        return dcm.pixel_array / max_val  # Normalize