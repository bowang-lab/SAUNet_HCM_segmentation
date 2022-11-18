import os
import re
import functools
import fnmatch
import numpy as np
#from keras import utils as kutil

def find_recursive(root_dir, ext='.jpg'):
    files = []
    for root, dirnames, filenames in os.walk(root_dir):
        for filename in fnmatch.filter(filenames, '*' + ext):
            files.append(os.path.join(root, filename))
    return files


class AverageMeter(object):
    """Computes and stores the average and current value"""
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


def unique(ar, return_index=False, return_inverse=False, return_counts=False):
    ar = np.asanyarray(ar).flatten()

    optional_indices = return_index or return_inverse
    optional_returns = optional_indices or return_counts

    if ar.size == 0:
        if not optional_returns:
            ret = ar
        else:
            ret = (ar,)
            if return_index:
                ret += (np.empty(0, np.bool),)
            if return_inverse:
                ret += (np.empty(0, np.bool),)
            if return_counts:
                ret += (np.empty(0, np.intp),)
        return ret
    if optional_indices:
        perm = ar.argsort(kind='mergesort' if return_index else 'quicksort')
        aux = ar[perm]
    else:
        ar.sort()
        aux = ar
    flag = np.concatenate(([True], aux[1:] != aux[:-1]))

    if not optional_returns:
        ret = aux[flag]
    else:
        ret = (aux[flag],)
        if return_index:
            ret += (perm[flag],)
        if return_inverse:
            iflag = np.cumsum(flag) - 1
            inv_idx = np.empty(ar.shape, dtype=np.intp)
            inv_idx[perm] = iflag
            ret += (inv_idx,)
        if return_counts:
            idx = np.concatenate(np.nonzero(flag) + ([ar.size],))
            ret += (np.diff(idx),)
    return ret


def colorEncode(labelmap, colors, mode='BGR'):
    labelmap = labelmap.astype('int')
    labelmap_rgb = np.zeros((labelmap.shape[0], labelmap.shape[1], 3),
                            dtype=np.uint8)
    for label in unique(labelmap):
        if label < 0:
            continue
        labelmap_rgb += (labelmap == label)[:, :, np.newaxis] * \
            np.tile(colors[label],
                    (labelmap.shape[0], labelmap.shape[1], 1))

    if mode == 'BGR':
        return labelmap_rgb[:, :, ::-1]
    else:
        return labelmap_rgb


def accuracy(preds, label):
    valid = (label >= 1)
    acc_sum = (valid * (preds == label)).sum()
    valid_sum = valid.sum()
    acc = float(acc_sum) / (valid_sum + 1e-10)
    return acc, valid_sum


def intersectionAndUnion(imPred, imLab, numClass):
    imPred = np.asarray(imPred).copy()
    imLab = np.asarray(imLab).copy()

    imPred += 1
    imLab += 1
    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    imPred = imPred * (imLab > 0)

    # Compute area intersection:
    intersection = imPred * (imPred == imLab)
    (area_intersection, _) = np.histogram(
        intersection, bins=numClass, range=(1, numClass))

    # Compute area union:
    (area_pred, _) = np.histogram(imPred, bins=numClass, range=(1, numClass))
    (area_lab, _) = np.histogram(imLab, bins=numClass, range=(1, numClass))
    area_union = area_pred + area_lab - area_intersection
    #print("I: " + str(area_intersection))
    #print("U: " + str(area_union))
    return (area_intersection, area_union)

def Dice(pred, lab, numClass):
    dice = [-1] #put first element as -1 as 0th index class is background and we do not care.
    for i in range(1, numClass):
        x = (pred == i)
        x_count = np.count_nonzero(x)
        #print('x shape: '+str(type(x))+' '+str(x.shape)+' '+str(x.min())+' '+str(x.max()))
        y = (lab == i)
        y_count = np.count_nonzero(y)
        #print('y shape: '+str(type(y))+' '+str(y.shape)+' '+str(y.min())+' '+str(x.max()))
        intersection = x * y
        #print('intersection shape: '+str(type(intersection))+' '+str(intersection.shape)+' '+str(intersection.max())+' '+str(intersection.min()))
        d = (2*np.count_nonzero(intersection))/(np.count_nonzero(x)+np.count_nonzero(y)+1E-10)
        #print('d_tmp:'+ str(np.count_nonzero(x))+' '+str(np.count_nonzero(y))+' '+str(np.count_nonzero(intersection)))
        #d = (2*intersection)/(x.sum() + y.sum() + 1E-10)
        #print('d shape: '+str(type(d))+' '+str(d.shape))
        dice.append(d)
        #print('dice shape: '+str(type(dice))+' '+str(len(dice)))
    return dice
'''
def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]
'''

def to_categorical(y, num_classes):
    #print('to categorical y shape: '+str(y.shape))
    binary_matrix = list()
    for i in range(0, num_classes):
        x = (y == i)
        #print('x shape: '+str(x.shape))
        binary_matrix.append(x.astype(int))
    return np.array(binary_matrix)

def newDiceBoston(y_true, y_pred):
    smooth = 1e-5
    y_true_f = np.ndarray.flatten(y_true)
    y_pred_f = np.ndarray.flatten(y_pred)
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection ) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def newDiceCoefEvalBoston(y_true, y_pred, num_class):
    #print('y_pred '+str(y_pred.shape)+' '+str(np.max(y_pred))+' '+str(np.min(y_pred)))
    #print('y_true '+str(y_true.shape)+' '+str(np.max(y_true))+' '+str(np.min(y_true)))
    #y_labels = y_pred.argmax(axis=-1)
    #print('y_labels'+str(y_labels.shape)+' '+str(np.max(y_labels))+' '+str(np.min(y_labels)))
    y_pred_dum =  to_categorical(y_pred, num_classes = num_class)
    y_true_dum =  to_categorical(y_true[0], num_classes = num_class)
    #print('y_pred_dum '+str(y_pred_dum.shape)+' '+str(np.max(y_pred_dum))+' '+str(np.min(y_pred_dum)))
    #print('y_true_dum '+str(y_true_dum.shape)+' '+str(np.max(y_true_dum))+' '+str(np.min(y_true_dum)))
    
    y_back = y_pred_dum[:,:,0]
    #print('y_back '+str(y_back.shape)+' '+str(np.max(y_back))+' '+str(np.min(y_back)))
    y_epi = y_pred_dum[:,:,1]
    #print('y_epi '+str(y_epi.shape)+' '+str(np.max(y_epi))+' '+str(np.min(y_epi)))
    y_endo = y_pred_dum[:,:,2]
    #print('y_endo '+str(y_endo.shape)+' '+str(np.max(y_endo))+' '+str(np.min(y_endo)))
    y_scr = y_pred_dum[:,:,-1]
    #print('y_scr '+str(y_scr.shape)+' '+str(np.max(y_scr))+' '+str(np.min(y_scr)))
    
    #print('y_true_dum '+str(y_true_dum.shape)+' '+str(np.max(y_true_dum))+' '+str(np.min(y_true_dum)))
    
    dice_back = newDiceBoston(y_true_dum[:,:,0], y_back)
    dice_epi = newDiceBoston(y_true_dum[:,:,1], y_epi)
    dice_endo = newDiceBoston(y_true_dum[:,:,2], y_endo)
    dice_scr = newDiceBoston(y_true_dum[:,:,-1], y_scr)
    #print('dice scores shapes: '+str(dice_epi.shape)+' '+str(dice_endo.shape)+' '+str(dice_scr.shape))
    
    return [dice_back, dice_epi, dice_endo, dice_scr]


def get_TP_TN_FP_FN(y_true, y_pred):
    smooth = 1e-5
    #print('get_TP_TN_FP_FN input size: '+str(y_true.shape)+' '+str(y_pred.shape))
    y_true_f = np.ndarray.flatten(y_true)
    y_pred_f = np.ndarray.flatten(y_pred)
    #print('max y_true_f: '+str(np.max(y_true_f))+str(y_true_f.shape))
    #print('min y_true_f: '+str(np.min(y_true_f)))
    #print('max y_pred_f: '+str(np.max(y_pred_f))+str(y_pred_f.shape))
    #print('min y_pred_f: '+str(np.min(y_pred_f)))
    TP = np.sum(y_true_f * y_pred_f)
    FP = np.sum(y_pred_f - (y_true_f * y_pred_f))
    TN = np.sum((y_true_f-1) * (y_pred_f-1))
    FN = np.sum(y_true_f - (y_pred_f*y_true_f))
    '''
    print('TP: '+str(TP)+' '+str(type(TP)))
    print('FP: '+str(FP)+' '+str(type(FP)))
    print('TN: '+str(TN)+' '+str(type(TN)))
    print('FN: '+str(FN)+' '+str(type(FN)))
    print('ALL: '+str(TP+TN+FP+FN))
    '''
    return [TP, FP, TN, FN]



def get_confusion_matrix(y_pred, y_true, num_class):
     
    y_pred_dum =  to_categorical(y_pred, num_classes = num_class)
    y_true_dum =  to_categorical(y_true[0], num_classes = num_class)
    #print('y_pred_dum '+str(y_pred_dum.shape)+' '+str(np.max(y_pred_dum))+' '+str(np.min(y_pred_dum)))
    #print('y_true_dum '+str(y_true_dum.shape)+' '+str(np.max(y_true_dum))+' '+str(np.min(y_true_dum)))
    '''
    y_back = y_pred_dum[:,:,0]
    y_epi = y_pred_dum[:,:,1]
    y_endo = y_pred_dum[:,:,2]
    y_scr = y_pred_dum[:,:,-1]
    '''

    y_back = y_pred_dum[0,:,:]
    y_epi = y_pred_dum[1,:,:]
    y_endo = y_pred_dum[2,:,:]
    y_scr = y_pred_dum[-1,:,:]

    back = get_TP_TN_FP_FN(y_true_dum[0,:,:], y_back)
    epi = get_TP_TN_FP_FN(y_true_dum[1,:,:], y_epi)
    endo = get_TP_TN_FP_FN(y_true_dum[2,:,:], y_endo)
    scr = get_TP_TN_FP_FN(y_true_dum[-1,:,:], y_scr)

    '''
    back = get_TP_TN_FP_FN(y_true_dum[:,:,0], y_back)
    epi = get_TP_TN_FP_FN(y_true_dum[:,:,1], y_epi)
    endo = get_TP_TN_FP_FN(y_true_dum[:,:,2], y_endo)
    scr = get_TP_TN_FP_FN(y_true_dum[:,:,-1], y_scr)
    '''
    #print('confusion matrix:')
    #print([back, epi, endo, scr])
    return [back, epi, endo, scr]

class NotSupportedCliException(Exception):
    pass


def process_range(xpu, inp):
    start, end = map(int, inp)
    if start > end:
        end, start = start, end
    return map(lambda x: '{}{}'.format(xpu, x), range(start, end+1))


REGEX = [
    (re.compile(r'^gpu(\d+)$'), lambda x: ['gpu%s' % x[0]]),
    (re.compile(r'^(\d+)$'), lambda x: ['gpu%s' % x[0]]),
    (re.compile(r'^gpu(\d+)-(?:gpu)?(\d+)$'),
     functools.partial(process_range, 'gpu')),
    (re.compile(r'^(\d+)-(\d+)$'),
     functools.partial(process_range, 'gpu')),
]


def parse_devices(input_devices):

    """Parse user's devices input str to standard format.
    e.g. [gpu0, gpu1, ...]

    """
    ret = []
    for d in input_devices.split(','):
        for regex, func in REGEX:
            m = regex.match(d.lower().strip())
            if m:
                tmp = func(m.groups())
                # prevent duplicate
                for x in tmp:
                    if x not in ret:
                        ret.append(x)
                break
        else:
            raise NotSupportedCliException(
                'Can not recognize device: "{}"'.format(d))
    return ret
'''
def torch_pixelwise_gradient(f, *varargs, **kwargs):
    N = len(f.shape)
    n = len(varargs)

    if n == 0:
        dx = torch.tensor([1.0])*N
    elif n == 1:
        dx = torch.tensor([varargs[0]])*N
    elif n == N:
        dx = torch.tensor(varargs)
    else:
        raise SyntaxError(
                "invalid number of arguments")

    edge_order = kwargs.pop('edge_order', 1)
    if kwargs:
        raise TypeError('"{}" are not valid keyword arguments.'.format(
                                                    '", "'.join(kwargs.keys())))
    if edge_order > 2:
        raise ValueError("'edge_order' greater than 2 not supported")

    outvals = []

    slice1 = [slice(None)]*N
    slice2 = [slice(None)]*N
    slice3 = [slice(None)]*N
    slice4 = [slice(None)]*N
    
    y = f.float()

    for axis in range(N):
        if y.shape[axis] < 2:
            raise ValueError(
                    "Shape of array too small to calculate a numerical gradient, "
                    "at least two elements are required.")

        if y.shape[axis] == 2 or edge_order == 1:
            out = f.new(x.size()).float()

            uniform_spacing = 
'''
