import pydicom
import os.path
import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn as nn
from torch.utils import data
from .augmentations import Compose, RandomRotate, PaddingCenterCrop
import random
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.morphology import distance_transform_edt
import scipy.io


class GaussianBlur(object):
    def __init__(self, min=0.1, max=2.0, kernel_size=15):
        self.min = min
        self.max = max
        self.kernel_size = kernel_size

    def __call__(self, sample):
        sample = np.array(sample)
        prob = np.random.random_sample()

        sigma = (self.max-self.min) * np.random.random_sample() + self.min
        sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)

        return sample


def augment_gamma(data_sample, gamma_range=(0.5, 2), invert_image=False, epsilon=1e-7, per_channel=False,
                  retain_stats=False):
    if invert_image:
        data_sample = - data_sample
    if not per_channel:
        if retain_stats:
            mn = data_sample.mean()
            sd = data_sample.std()
        if np.random.random() < 0.5 and gamma_range[0] < 1:
            gamma = np.random.uniform(gamma_range[0], 1)
        else:
            gamma = np.random.uniform(max(gamma_range[0], 1), gamma_range[1])
        minm = data_sample.min()
        rnge = data_sample.max() - minm
        data_sample = np.power(((data_sample - minm) / float(rnge + epsilon)), gamma) * rnge + minm
        if retain_stats:
            data_sample = data_sample - data_sample.mean() + mn
            data_sample = data_sample / (data_sample.std() + 1e-8) * sd
    else:
        for c in range(data_sample.shape[0]):
            if retain_stats:
                mn = data_sample[c].mean()
                sd = data_sample[c].std()
            if np.random.random() < 0.5 and gamma_range[0] < 1:
                gamma = np.random.uniform(gamma_range[0], 1)
            else:
                gamma = np.random.uniform(max(gamma_range[0], 1), gamma_range[1])
            minm = data_sample[c].min()
            rnge = data_sample[c].max() - minm
            data_sample[c] = np.power(((data_sample[c] - minm) / float(rnge + epsilon)), gamma) * float(rnge + epsilon) + minm
            if retain_stats:
                data_sample[c] = data_sample[c] - data_sample[c].mean() + mn
                data_sample[c] = data_sample[c] / (data_sample[c].std() + 1e-8) * sd
    if invert_image:
        data_sample = - data_sample
    return data_sample


class Dataloader2D(data.Dataset):

    def __init__(self,
                 split='train',
                 augmentations=None,
                 k=5,
                 k_split=1,
                 target_size=(256, 256)
                 ):
        self.target_size = target_size
        self.split = split
        self.k = k
        self.root = '/cluster/projects/bwanggroup/HCM_MRI/shared/data/CLEAN_DATA/ALL'
        self.k_split = int(k_split)
        self.augmentations = augmentations
        self.moments = self.get_moments()
        self.rlist = self.remove_list()
        self._imgs, self._c1, self._c2, self._c = self.get_data(self.root+"/old_new_dicom_Nov27.txt",
                                                                self.root+"/old_new_con1.txt",
                                                                self.root+"/old_new_con2.txt",
                                                                self.root+"/old_new_con_Nov27.txt")
        
        self.imgs, self.c1, self.c2, self.c = [], [], [], []
        self.data = []
        self.make_partition()
            
        self.res = {}

        self.parse_res()

        self.delta = {}
        self.get_delta()

    def __len__(self):
        length = 0
        if self.split =='val_external':
            length = 34
        else:
            length = len(self.imgs)
        return length

    def get_delta(self):
        with open('/cluster/projects/bwanggroup/HCM_MRI/shared/code/data/deltas.txt', 'r') as f:
            for line in f:
                idx, d = line.split('\t')
                self.delta[idx] = float(d[:-1])

    def make_partition(self):
        patient = ['007-271', '007-259', '007-033', '007-100', '007-056', '007-004', '007-145', '007-308', '007-124', '007-143', '007-129', '007-017', '007-309', '007-121', '007-312', '007-024', '007-031', '007-045', '007-018', '007-138', '007-274', '007-097', '007-139', '007-114', '007-020', '007-112', '007-119', '007-347', '007-093', '007-028', '007-233', '007-035', '007-323', '007-287', '007-116', '007-268', '007-092', '007-006', '007-019', '007-125', '007-111', '007-319', '007-095', '007-088', '007-104', '007-007', '007-041', '007-352', '007-247', '007-354', '007-224', '007-059', '007-135', '007-202', '007-137', '007-052', '007-126', '007-332', '007-346', '007-338', '007-356', '007-147', '007-331', '007-074', '007-094', '007-209', '007-232', '007-234', '007-136', '007-339', '007-016', '007-325', '007-049', '007-057', '007-036', '007-022', '007-025', '007-046', '007-349', '007-013', '007-055', '007-334', '007-140', '007-075', '007-335', '007-317', '007-158', '007-037', '007-217', '007-127', '007-117', '007-063', '007-144', '007-034', '007-350', '007-113', '007-330', '007-344', '007-345', '007-032', '007-082', '007-311', '007-068', '007-146', '007-142', '007-355', '007-151', '007-047', '007-333', '007-351', '007-153', '007-336', '007-021', '007-164', '007-122', '007-236', '007-058', '007-156', '007-103', '007-048', '007-030', '007-042', '007-071', '007-123', '007-085', '007-324', '007-011', '007-320', '007-141', '007-077', '007-076', '007-315', '007-132', '007-062', '007-155', '007-099', '007-150', '007-090', '007-115', '007-342', '007-043', '007-157', '007-310', '007-343', '007-023', '007-039', '007-348', '007-194', '007-108', '002-597', '002-612', '002-607', '002-606', '002-602', '002-608', '002-622', '002-620', '002-617', '002-614', '002-610', '002-615', '002-599', '002-601', '002-609', '002-603', '002-600', '002-604', '002-621', '002-623', '002-595', '002-605', '002-624', '002-618', '002-616', '002-594', '002-591', '002-598', '002-438', '002-453', '002-451', '002-445', '002-439', '002-447', '002-581', '002-484', '002-546', '002-470', '002-522', '002-532', '002-555', '002-505', '002-485', '002-566', '002-520', '002-538', '002-514', '002-476', '002-585', '002-544', '002-501', '002-499', '002-512', '002-561', '002-575', '002-503', '002-557', '002-580', '002-571', '002-572', '002-559', '002-516', '002-508', '002-498', '002-531', '002-579', '002-492', '002-539', '002-515', '002-584', '002-582', '002-482', '002-513', '002-574', '002-558', '002-491', '002-562', '002-587', '002-473', '002-545', '002-550', '002-553', '002-576', '002-477', '002-547', '002-468', '002-502', '002-487', '002-511', '002-471', '002-564', '002-525', '002-589', '002-507', '002-548', '002-542', '002-527', '002-535', '002-478', '002-488', '002-552', 't110', 't120', 't128', 't115', 't127', 't106', 't114', 't125', 't129', 't111', 't104', 't117', 't122', 't119', 't108', 't116', 't124', 't103', 't105', 't107', 't130', 't112', 't118', 't123', 't109', '002-504', '007-318', '007-154', '007-029', '007-110', '007-061', '007-040', '007-133', '007-326', '007-288', '007-102', '007-328', '007-053', '007-098', '007-044', '007-131', '007-313', '007-012', '007-005', '007-010', '007-089', '007-357', '007-107', '007-038', '007-106', '007-083', '007-256', '007-027', '007-120', '007-213', '007-341', '007-009', '007-149', '007-316', '007-183', '007-069', '007-067', '002-530', '002-497', '002-510', '002-495', '002-560', '002-528', '002-569', '002-568', '002-469', '002-570', '002-479', '002-494', '002-486', '002-567', '002-536', '002-474', '002-549', '002-540', '002-543', '002-554', '002-551', '002-475', '002-493', '002-518', '002-583', '002-519', '002-565', '002-521', '002-529', 't101', 't113', 't126', 't102', 't121']


        for i in range(len(self._imgs)):
            part = self._imgs[i].split('/')
            identity = ''
            if '007-' in part[6]:
                identity = part[6]

            elif '002-' in part[7]:
                identity = part[7]

            elif '002-' in part[6]:
                identity = part[6]

            elif 'anonymized_' in self._imgs[i]:
                identity = part[-3][:-1]

            tmp = patient.index(identity)%5+1
            if identity == '007-104':
                continue
            if self.split == 'train' and tmp != self.k_split:
                self.imgs.append(self._imgs[i])
                self.c1.append(self._c1[i])
                self.c2.append(self._c2[i])
                self.c.append(self._c[i])


            if self.split == 'val' and tmp == self.k_split:
                self.imgs.append(self._imgs[i])
                self.c1.append(self._c1[i])
                self.c2.append(self._c2[i])
                self.c.append(self._c[i])

        return


    def get_occurence_count(self, data, sample):
        counter = 0
        for d in data:
            if sample in d:
                counter += 1

        return counter

    def parse_res(self):
        with open("/cluster/projects/bwanggroup/HCM_MRI/shared/code/data/res.txt", "r") as f:
            for line in f:
                if(line.split(' ') == 3):
                    tag, x, y = line.split(' ')
                    r = (float(x), float(y[:-1]))
                    self.res["0"+tag] = r
                else:
                    x, y = line.split(' ')[-2], line.split(' ')[-1]
                    tag = ""
                    for i in range(len(line.split(' ')) - 2):
                        tag += line.split(' ')[i] + " "
                    tag = tag[:-1]
                    r = (float(x), float(y[:-1]))
                    self.res["0"+tag] = r


    def myStandardize(self, img):
        if len(np.unique(img)) == 3:
            img[img == 1] = 3
            img[img == 2] = 1
            img[img == 3] = 2
        elif len(np.unique(img)) == 4:
            img[img == 1] = 4
            img[img == 2] = 1
            img[img == 4] = 2
        return img


    def __getitem__(self, i):

        if self.split =='val_external':
            img, seg, fname = self.get_Boston_sample_data(i)
        else:
            self.imgs[i] = self.imgs[i].rstrip()
            parts = self.imgs[i].split("/")
            img, seg, fname = self.get_dicom_contour(self.imgs[i], self.c[i].rstrip(), self.c1[i], self.c2[i])

        img -= img.min()
        orig = img.copy().astype(np.int32)
        img, seg = self.augmentations(img.astype(np.uint32), seg.astype(np.uint8))

        
        if self.split =='train' and random.uniform(0, 1.0) <= 0.5:
            if img.ndim == 2:
                img = np.expand_dims(img, axis=2)
            seg = np.expand_dims(seg, axis=2)
            stacked = np.concatenate((img, seg), axis=2)
            red = self.random_elastic_deformation(stacked, alpha=500, sigma=20).transpose(2,0,1)
            img, seg = red[0], red[1]
        
        img = (img - img.mean())/img.std()

        if img.ndim == 2:
            img = np.expand_dims(img, axis=0)
            img = np.concatenate((img, img, img), axis=0)

        img = torch.from_numpy(img).float()
        seg = torch.from_numpy(seg).long()
        edges = self.mask_to_edges(seg)
        orig = torch.from_numpy(orig)

        try:
            r = self.res[fname]
            scale = r[0]/float(orig.shape[-2]) * r[1]/float(orig.shape[-1])
        except:
            scale = 1

        if self.split =='val_external':
             identity = 'Boston sample'
        else:
            identity = ''
            if '007-' in parts[6]:
                identity = parts[6]

            elif '002-' in parts[7]:
                identity = parts[7]

            elif '002-' in parts[6]:
                identity = parts[6]

            elif 'anonymized' in self.imgs[i]:
                identity = parts[10][:-1]
        
	try:
            delta = self.delta[fname]
        except:
            delta = 0 
        return img, (seg, edges)#, fname, orig, scale, identity# delta, fname


    def get_moments(self):
        d = {}
        s = []
        with open("/cluster/projects/bwanggroup/HCM_MRI/shared/code/data/Patient_normal_stats.txt", 'r') as f:
            for line in f:
                splitted = line.split(' ')
                if len(splitted) == 3:
                    idx, mu, std = line.split(' ')
                else:
                     idx = ""
                     for i in range(len(splitted)-2):
                         idx += " " + splitted[i] if idx != "" else splitted[i]
                     mu, std = splitted[-2], splitted[-1]

                d[idx] = (float(mu), float(std[:-1]))
        
        return d


    def remove_list(self):
        s = []
        with open("/cluster/projects/bwanggroup/HCM_MRI/shared/code/data/remove.txt", 'r') as f:
            for l in f:
                s.append(l[:-1])
        return s


    def get_dicom_contour(self, dicom, contour, line1, line2):

        con1 = []
        con2 = []

        filename = dicom

        with open(contour, 'r') as fp:
            x1, y1 = line1.split(',')
            x2, y2 = line2.split(',')

            if 'anonymized' in filename:
                for i, line in enumerate(fp):
                    if i >= int(x1) and i <= int(y1):
                        part = line.split(',')
                        con1.append([int(float(part[0])), int(float(part[1]))])
                    if i >= int(x2) and i <= int(y2):
                        part = line.split(',')
                        con2.append([int(float(part[0])), int(float(part[1]))])
            else:
                for i, line in enumerate(fp):
                    if i >= int(x1)-1 and i < int(y1[:-1]):
                        con1.append([int(float(j)) for j in line.split()])
                    if i >= int(x2)-1 and i < int(y2[:-1]):
                        con2.append([int(float(j)) for j in line.split()])
            fp.close()

        # new Toronto samples
        if 'anonymized' in filename:
            d = Image.open(filename) 
            pix = d.load()
            dic = np.zeros((256, 256), dtype = "uint8")

            for x in range(0,256):
                for y in range(0,256):
                    dic[x][y] = pix[x,y][0]*0.2989+pix[x,y][1]*0.5870+pix[x,y][2]*0.1140
            dicom = np.array(dic)
        else: # other samples
            dicom = pydicom.dcmread(filename).pixel_array

         
        con = np.zeros(dicom.shape)
        cv2.fillPoly(con, [np.array(con2, dtype='int32')], 1)
        cv2.fillPoly(con, [np.array(con1, dtype='int32')], 2)


        if 'anonymized' in filename:
            for c in con1:
                con[c[1], c[0]] = 1
        
	fname = ''
        try:
            if 'anonymized' in filename:
                fname = filename.split('/')[-3][:-1]
                mu, std = self.moments.get(fname)               
            else:
                if '002-438 to 002-467 Tufts CMR ECG substudy' in filename:
                    fname = filename.split('/')[7]
                else:
                    fname = filename.split('/')[6]
                mu, std = self.moments.get(fname)
        except:
            mu, std = 10000., 10000.
            
        #get valid area between endocardium & epicardium
        outer = np.zeros(dicom.shape)
        inner = np.zeros(dicom.shape)
        cv2.fillPoly(outer, [np.array(con2)], 1)
        cv2.fillPoly(inner, [np.array(con1)], 1)


        if 'anonymized' in filename:
            for c in con1:
                inner[c[1], c[0]] = 0
        diff = (outer - inner) * dicom # valid region
        #con[diff >= mu + 4*std] = 3
        con[diff >= mu + 6*std] = 3 
        
        ## remove scars close to epicardium
        pre = con.copy()

        tmp = np.zeros_like(con)
        z = np.zeros_like(tmp)
        r = 2
        tmp[r:, r:] = pre[:-r, :-r]
        tmp = (tmp == 3) & (pre == 0)
        z[:-r, :-r] = tmp[r:, r:]
        con[z == True] = 1

        tmp = np.zeros_like(con)
        z = np.zeros_like(tmp)
        tmp[:,r:] = pre[:, :-r]
        tmp = (tmp == 3) & (pre == 0)
        z[:, :-r] = tmp[:, r:]
        con[z == True] = 1

        tmp = np.zeros_like(con)
        z = np.zeros_like(tmp)
        tmp[:-r, r:] = pre[r:, r:]
        tmp = (tmp == 3) & (pre == 0)
        z[r:, r:] = tmp[:-r, r:]
        con[z == True] = 1

        tmp = np.zeros_like(con)
        z = np.zeros_like(tmp)
        tmp[:-r, :] = pre[r:, :]
        tmp = (tmp == 3) & (pre == 0)
        z[r:, :] = tmp[:-r, :]
        con[z == True] = 1

        tmp = np.zeros_like(con)
        z = np.zeros_like(tmp)
        tmp[:-r, :-r] = pre[r:, r:]
        tmp = (tmp == 3) & (pre == 0)
        z[r:, r:] = tmp[:-r, :-r]
        con[z == True] = 1

        tmp = np.zeros_like(con)
        z = np.zeros_like(tmp)
        tmp[:, :-r] = pre[:, r:]
        tmp = (tmp == 3) & (pre == 0)
        z[:, r:] = tmp[:, :-r]
        con[z == True] = 1

        tmp = np.zeros_like(con)
        z = np.zeros_like(tmp)
        tmp[:-r, :] = pre[r:, :]
        tmp = (tmp == 3) & (pre == 0)
        z[r:, :] = tmp[:-r, :]
        con[z == True] = 1

        tmp = np.zeros_like(con)
        z = np.zeros_like(tmp)
        tmp[r:, :-r] = pre[:-r, r:]
        tmp = (tmp == 3) & (pre == 0)
        z[:-r, r:] = tmp[r:, :-r]
        con[z == True] = 1

        ## remove scars close to endocardium
        pre = con.copy()

        tmp = np.zeros_like(con)
        z = np.zeros_like(tmp)
        r = 1
        tmp[r:, r:] = pre[:-r, :-r]
        tmp = (tmp == 3) & (pre == 2)
        z[:-r, :-r] = tmp[r:, r:]
        con[z == True] = 2

        tmp = np.zeros_like(con)
        z = np.zeros_like(tmp)
        tmp[:,r:] = pre[:, :-r]
        tmp = (tmp == 3) & (pre == 2)
        z[:, :-r] = tmp[:, r:]
        con[z == True] = 2

        tmp = np.zeros_like(con)
        z = np.zeros_like(tmp)
        tmp[:-r, r:] = pre[r:, r:]
        tmp = (tmp == 3) & (pre == 2)
        z[r:, r:] = tmp[:-r, r:]
        con[z == True] = 2

        tmp = np.zeros_like(con)
        z = np.zeros_like(tmp)
        tmp[:-r, :] = pre[r:, :]
        tmp = (tmp == 3) & (pre == 2)
        z[r:, :] = tmp[:-r, :]
        con[z == True] = 2

        tmp = np.zeros_like(con)
        z = np.zeros_like(tmp)
        tmp[:-r, :-r] = pre[r:, r:]
        tmp = (tmp == 3) & (pre == 2)
        z[r:, r:] = tmp[:-r, :-r]
        con[z == True] = 2

        tmp = np.zeros_like(con)
        z = np.zeros_like(tmp)
        tmp[:, :-r] = pre[:, r:]
        tmp = (tmp == 3) & (pre == 2)
        z[:, r:] = tmp[:, :-r]
        con[z == True] = 2

        tmp = np.zeros_like(con)
        z = np.zeros_like(tmp)
        tmp[:-r, :] = pre[r:, :]
        tmp = (tmp == 3) & (pre == 2)
        z[r:, :] = tmp[:-r, :]
        con[z == True] = 2

        tmp = np.zeros_like(con)
        z = np.zeros_like(tmp)
        tmp[r:, :-r] = pre[:-r, r:]
        tmp = (tmp == 3) & (pre == 2)
        z[:-r, r:] = tmp[r:, :-r]
        con[z == True] = 2

        # remove small scars
        temp = (con==3).astype(np.uint8)
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(temp, connectivity=8)

        sizes = stats[1:, -1]; nb_components = nb_components - 1
        min_size = 2

        out = np.zeros((temp.shape))
        for i in range(0, nb_components):
            if sizes[i] >= min_size:
                out[output == i + 1] = 1

        con[(temp - out) > 0] = 1

        return dicom, con, fname

    def get_data(self, dicom_list, cont1_list, cont2_list, con_list):

        dicom = []
        con1 = []
        con2 = []
        con = []

        with open(dicom_list) as f:
            dicom = f.readlines()

        with open(cont1_list) as f:
            con1 = f.readlines()

        with open(cont2_list) as f:
            con2 = f.readlines()

        with open(con_list) as f:
            con = f.readlines()
        

        d = []
        c1 = []
        c2 = []
        c = []

        for i, line in enumerate(dicom):
            num = line.split('/')[6]
            if 'anonymized' in line:
                num = line.split('/')[-3][:-1]
            if num not in self.rlist:
                d.append(dicom[i])
                c1.append(con1[i])
                c2.append(con2[i])
                c.append(con[i])

        return d, c1, c2, c


    def get_Boston_sample_data(self, index):
    
        directory = '/cluster/projects/bwanggroup/HCM_MRI/shared/data/Boston2021Data/'
        file = ['12_comboBI_pLGE_05_image', 
                '43_comboBI_pLGE_00_image', 
                '104_comboBI_pLGE_08_image']

        dicom = list()
        label_list = list()
        fname = list()

        for f in file:
            annots_img = scipy.io.loadmat(directory+f+".mat")
            annots_label = scipy.io.loadmat(directory+f+"_label.mat")

            for i in range(annots_img['tmp_vol_im'].astype(np.float).shape[3]):

                label = annots_label['tmp_vol_im'].astype(np.float)[:,:,0,i]
                if len(np.unique(label)) == 3:
                    label[label == 1] = 3
                    label[label == 2] = 1
                    label[label == 3] = 2
                elif len(np.unique(label)) == 4:
                    label[label == 1] = 4
                    label[label == 2] = 1
                    label[label == 4] = 2
                label_std = label

                dic = np.zeros((160, 160), dtype = "uint8")
                pix = annots_img['tmp_vol_im'].astype(np.float)[:,:,:,i]
                for x in range(0,160):
                    for y in range(0,160):
                        dic[x][y] = pix[x,y][0]*0.2989+pix[x,y][1]*0.5870+pix[x,y][2]*0.1140
                img = np.array(dic)
                
                if not(len(np.unique(label_std)) == 1 or np.isnan(np.unique(label_std)).any()):

                    dicom.append(img)
                    label_list.append(label_std)
                    fname.append(f+"_"+str(i))

        return np.array(dicom)[index], np.array(label_list)[index], np.array(fname)[index]

        
    def random_elastic_deformation(self, image, alpha, sigma, mode='nearest',
                                   random_state=None):
        """Elastic deformation of images as described in [Simard2003]_.
    ..  [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
        Convolutional Neural Networks applied to Visual Document Analysis", in
        Proc. of the International Conference on Document Analysis and
        Recognition, 2003.
        """
        assert len(image.shape) == 3

        if random_state is None:
            random_state = np.random.RandomState(None)

        height, width, channels = image.shape

        dx = gaussian_filter(2*random_state.rand(height, width) - 1,
                         sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter(2*random_state.rand(height, width) - 1,
                         sigma, mode="constant", cval=0) * alpha

        x, y = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        indices = (np.repeat(np.ravel(x+dx), channels),
                np.repeat(np.ravel(y+dy), channels),
                np.tile(np.arange(channels), height*width))

        values = map_coordinates(image, indices, order=1, mode=mode)

        return values.reshape((height, width, channels))

    def mask_to_onehot(self, mask, num_classes=3):
        _mask = [mask == i for i in range(1, num_classes+1)]
        _mask = [np.expand_dims(x, 0) for x in _mask]
        return np.concatenate(_mask, 0)

    def onehot_to_binary_edges(self, mask, radius=2, num_classes=3):
        if radius < 0:
            return mask

        # We need to pad the borders for boundary conditions
        mask_pad = np.pad(mask, ((0, 0), (1, 1), (1, 1)), mode='constant', constant_values=0)

        edgemap = np.zeros(mask.shape[1:])

        for i in range(num_classes):
            dist = distance_transform_edt(mask_pad[i, :])+distance_transform_edt(1.0-mask_pad[i, :])
            dist = dist[1:-1, 1:-1]
            dist[dist > radius] = 0
            edgemap += dist
        edgemap = np.expand_dims(edgemap, axis=0)
        edgemap = (edgemap > 0).astype(np.uint8)
        return edgemap

    def mask_to_edges(self, mask):
        _edge = mask
        _edge = self.mask_to_onehot(_edge)
        _edge = self.onehot_to_binary_edges(_edge)
        return torch.from_numpy(_edge).float()
