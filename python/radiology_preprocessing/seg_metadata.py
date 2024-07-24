from pydicom import dcmread
from pydicom_seg import SegmentReader
from scipy.ndimage import center_of_mass
from csv import DictReader
from os import listdir
from os.path import isfile
import numpy as np
import cv2

prefixNSCLC = '/home/sudipto/Templates/TCGA-NSCLC-manifest-1640349207402/'
prefixSegNSCLC = '/home/sudipto/Templates/TCGA-NSCLC-Seg-manifest-1695119508564/'


def read_series_sop_instance_ids(path: str):
    ids = []
    prefix = prefixNSCLC
    for fName in listdir(prefix + path):
        if isfile(prefix + path + fName) and fName[-4:] == '.dcm':
            ds = dcmread(prefix + path + fName)
            ids.append(ds.SOPInstanceUID)
    return ids


def print_segmentation_offsets() -> None:
    f = open('input/Radiology/seg-labels1.csv', 'r')
    reader = DictReader(f, delimiter=',')
    for row in reader:
        study_uid = row['Study UID']
        ct_file_location = row['File Location'][2:]
        seg_file_location = row['Segmentation File Location'][2:]
        if seg_file_location == '':
            print()
            continue

        ct_sop_ids = read_series_sop_instance_ids(ct_file_location + '/')
        dcm = dcmread(prefixSegNSCLC + seg_file_location + '/1-1.dcm')
        reader = SegmentReader()
        segment = reader.read(dcm)
        seg_sop_ids = segment.referenced_instance_uids
        if seg_sop_ids[0] in ct_sop_ids and seg_sop_ids[-1] in ct_sop_ids:
            print(ct_sop_ids.index(seg_sop_ids[0]), ct_sop_ids.index(seg_sop_ids[-1]))
        else:
            print()


def print_segmentation_bounds() -> None:
    f = open('input/Radiology/seg-labels1.csv', 'r')
    reader = DictReader(f, delimiter=',')
    for row in reader:
        study_uid = row['Study UID']
        ct_file_location = row['File Location'][2:]
        seg_file_location = row['Segmentation File Location'][2:]
        if seg_file_location == '':
            print(study_uid + '\t' + seg_file_location + '\t' + ct_file_location + '\t' +
                  '\t' + '\t' + '\t' + '\t' + '\t' + '\t' + '\t')
            continue

        # ct_array = read_series(ct_file_location + '/', (512, 512))
        # print(ct_array.shape)

        dcm = dcmread(prefixSegNSCLC + seg_file_location + '/1-1.dcm')
        reader = SegmentReader()
        segment = reader.read(dcm)
        seg_image = segment.segment_data(1)
        # print(seg_image.shape)

        segmented_slices = []
        ys = []
        xs = []
        ws = []
        hs = []
        for i in range(seg_image.shape[0]):
            if np.sum(seg_image[i, :, :]) > 0:
                segmented_slices.append(i)
            y, x, w, h = cv2.boundingRect(seg_image[i, :, :])
            ys.append(y)
            xs.append(x)
            ws.append(w)
            hs.append(h)
            # print(i, np.sum(seg_image[i, :, :]), x, y, h, w)
        cz, cx, cy = center_of_mass(seg_image)
        cz, cx, cy = int(cz), int(cx), int(cy)
        print(study_uid + '\t' + seg_file_location + '\t' + ct_file_location + '\t' +
              str(cz) + '\t' + str(cx) + '\t' + str(cy) + '\t' +
              str((np.max(segmented_slices) - np.min(segmented_slices))) + '\t' + str(np.max(hs)) + '\t' + str(np.max(ws)) + '\t' +
              str(np.min(segmented_slices)) + '\t' + str(np.max(segmented_slices)))


print_segmentation_offsets()

print_segmentation_bounds()
