from pydicom import dcmread
from pydicom_seg import SegmentReader
from scipy.ndimage import rotate, zoom, shift
from skimage.exposure import equalize_hist
from csv import DictReader
from copy import deepcopy
from os import listdir
from os.path import isfile
import matplotlib.pyplot as plt
import numpy as np

prefixNSCLC = '/home/sudipto/Templates/TCGA-NSCLC-manifest-1640349207402/'
prefixSegNSCLC = '/home/sudipto/Templates/TCGA-NSCLC-Seg-manifest-1695119508564/'


def zoom_volume_2d(vol: np.array, dim) -> np.array:
    # Resize across z-axis
    desired_width, desired_height = dim[1], dim[0]
    # print(desired_height, desired_width)
    current_width, current_height = vol.shape[1], vol.shape[0]
    width = current_width / desired_width
    height = current_height / desired_height
    width_factor, height_factor = (1 / width), (1 / height)
    volume = deepcopy(zoom(vol, (height_factor, width_factor), order=1))
    return volume


def read_series(path: str, shape, reverse: bool = False, order: str = 'DHW') -> np.array:
    """Read one volume"""
    slices = []
    prefix = prefixNSCLC
    for fName in listdir(prefix + path):
        if isfile(prefix + path + fName) and fName[-4:] == '.dcm':
            # if system() == 'Windows':
            #     ds = dcmread((prefix + path + fName).replace('/', '\\'))
            # else:
            #     ds = dcmread(prefix + path + fName)
            ds = dcmread(prefix + path + fName)

            if (ds.Rows, ds.Columns) == shape:
                slices.append(ds.pixel_array.astype("int32"))
    if reverse:
        slices.reverse()
    if order == 'HWD':
        volume = np.stack(slices, axis=2)
    elif order == 'DHW':
        volume = np.stack(slices, axis=0)
    else:
        raise ValueError('Error !!! Invalid \'order\' parameter, must be {HWD, DHW} ...')
    return volume


f = open('input/Radiology/seg-labels4.csv', 'r')
reader = DictReader(f, delimiter=',')
k = 0
for row in reader:
    if row['SegOffsetZ'] == '':
        print(k)
        k += 1
        continue
    ct_file_location = row['File Location'][2:]
    seg_file_location = row['Segmentation File Location'][2:]
    oz = int(row['SegOffsetZ'])
    num_rows, num_columns = int(row['Slice Rows']), int(row['Slice Columns'])
    cz, cx, cy = int(row['OriginZ']), int(row['OriginX']), int(row['OriginY'])
    low_x, high_x = cx - (int(row['SegSpanX']) // 2), cx + (int(row['SegSpanX']) // 2)
    low_y, high_y = cy - (int(row['SegSpanY']) // 2), cy + (int(row['SegSpanY']) // 2)

    ct_array = read_series(ct_file_location + '/', (num_rows, num_columns))
    # print(ct_array.shape)

    dcm = dcmread(prefixSegNSCLC + seg_file_location + '/1-1.dcm')
    seg_reader = SegmentReader()
    segment = seg_reader.read(dcm)
    seg_image = segment.segment_data(1)
    # print(seg_image.shape)

    images = [
        deepcopy(ct_array[cz + oz, :, :]), # in-order full CT
        deepcopy(ct_array[ct_array.shape[0] - (cz + oz), :, :]), # reverse full CT
        # deepcopy(equalize_hist(ct_array[cz + oz, :, :], nbins=256)), # in-order full CT with hist. eq.
        # deepcopy(equalize_hist(ct_array[ct_array.shape[0] - (cz + oz), :, :], nbins=256)), # reverse full CT with hist. eq.
        deepcopy(seg_image[cz, low_x:high_x, low_y:high_y]), # in-order segmentation mask
        deepcopy(seg_image[seg_image.shape[0] - cz, low_x:high_x, low_y:high_y]), # reverse segmentation mask
        deepcopy(ct_array[cz + oz, low_x:high_x, low_y:high_y]), # in-order segmented CT
        deepcopy(ct_array[ct_array.shape[0] - (cz + oz), low_x:high_x, low_y:high_y]) # reverse segmented CT
        # deepcopy(equalize_hist(ct_array[cz + oz, low_x:high_x, low_y:high_y], nbins=256)), # in-order segmented CT with hist. eq.
        # deepcopy(equalize_hist(ct_array[ct_array.shape[0] - (cz + oz), low_x:high_x, low_y:high_y], nbins=256))  # reverse segmented CT with hist. eq.
    ]

    # img = zoom_volume_2d(equalize_hist(ct_array[ct_array.shape[0] - (cz + oz), low_x:high_x, low_y:high_y], nbins=256), (128, 128))
    # images = [
    #     deepcopy(rotate(img, angle=-20)),
    #     deepcopy(rotate(img, angle=-10)),
    #     deepcopy(rotate(img, angle=10)),
    #     deepcopy(rotate(img, angle=20)),
    #     deepcopy(shift(img, (0, -20))),
    #     deepcopy(shift(img, (0, 20)))
    # ]

    print(k, cz, ct_array.shape[0] - cz, np.sum(seg_image[cz, :, :]), np.sum(seg_image[seg_image.shape[0] - cz, :, :]),
          images[0].shape, images[1].shape, images[2].shape, images[3].shape, images[4].shape, images[5].shape)
    fig, axs = plt.subplots(3, 2)
    for i, ax in enumerate(axs.flatten()):
        if i < len(images):
            ax.set_title(str(i))
            ax.imshow(images[i], cmap='gray')
        else:
            ax.remove()
    plt.show()

    print(k, ct_array.shape, seg_image.shape)
    # if k == 15:
    #     print(k, ct_array.shape, seg_image.shape, segment.spacing, segment.origin, segment.size)
    #     print(segment.segment_infos)
    #     print(segment.referenced_series_uid)
    #     print(segment.referenced_instance_uids)
    #     break

    k += 1
