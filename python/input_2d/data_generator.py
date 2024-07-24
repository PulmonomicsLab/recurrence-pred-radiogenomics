from typing import List, Tuple, Set
from os import listdir
from os.path import isfile
from csv import DictReader
from pydicom import dcmread
from keras.utils import Sequence
from tensorflow import expand_dims, device
from scipy.ndimage import rotate, zoom, shift
from skimage.exposure import equalize_hist
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from copy import deepcopy
import numpy as np

prefixNSCLC = '/home/sudipto/Templates/TCGA-NSCLC-manifest-1640349207402/'
prefixSegNSCLC = '/home/sudipto/Templates/TCGA-NSCLC-Seg-manifest-1695119508564/'

dev = '/cpu:0'


def crop_volume(volume: np.array, reverse: bool, centroid: Tuple[int, int, int],
                 x_span: Tuple[int, int], y_span: Tuple[int, int]) -> np.array:
    if reverse:
        cropped_volume = deepcopy(volume[volume.shape[0] - centroid[0], x_span[0]:x_span[1], y_span[0]:y_span[1]])
    else:
        cropped_volume = deepcopy(volume[centroid[0], x_span[0]:x_span[1], y_span[0]:y_span[1]])
    return cropped_volume


def zoom_volume_2d(vol: np.array, dim: Tuple[int, int]) -> np.array:
    # Resize across z-axis
    desired_width, desired_height = dim[1], dim[0]
    # print(desired_height, desired_width)
    current_width, current_height = vol.shape[1], vol.shape[0]
    width = current_width / desired_width
    height = current_height / desired_height
    width_factor, height_factor = (1 / width), (1 / height)
    volume = deepcopy(zoom(vol, (height_factor, width_factor), order=1))
    return volume


class DataGeneratorSegmentation2D(Sequence):
    def __init__(self, metadata_file_name: str, label_file_name: str, data_type: str = 'classification',
                 directory_field_name: str = 'File Location', pid_field_name: str = 'Subject ID',
                 dim: Tuple[int, int] = (128, 128), target_index_range: Tuple[int, int] = (1, 2),
                 rotation_angle: int = 0, shift_factor: int = 0, target_dtype: type = np.int64,
                 batch_size: int = 1, n_channels: int = 1, shuffle: bool = True,
                 scale_target: bool = False) -> None:
        """Constructor"""
        with device(dev):
            self.data_type = data_type
            self.dim = dim
            # self.order = order
            self.batch_size = batch_size
            self.rotation_angle = rotation_angle
            self.shift_factor = shift_factor
            self.shuffle = shuffle
            self.valid_pids = self._get_valid_pids(label_file_name)
            self.series_dirs, self.pids, self.shapes, self.centroids, self.x_spans, self.y_spans, self.reverses = \
                self._get_series_metadata(
                    metadata_file_name,
                    directory_field_name,
                    pid_field_name
                )
            self.targetLabels, self.targetNames = self._get_target_labels(
                label_file_name,
                target_index_range,
                target_dtype
            )
            if scale_target:
                self._scale_target_labels()
            self.indexes = np.arange(len(self.series_dirs))
            self.n_channels = n_channels
            # self.on_epoch_end()

    def __len__(self) -> int:
        """Denotes the number of batches per epoch"""
        with device(dev):
            return int(np.ceil(len(self.series_dirs) / self.batch_size))

    def __getitem__(self, index: int) -> Tuple[np.array, np.array]:
        """Generate one batch of data"""
        with device(dev):
            print('Loading Batch ' + str(index))
            indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
            # print(indexes)

            y = self.targetLabels[index * self.batch_size: (index + 1) * self.batch_size, :]

            volumes = []
            for i in indexes:
                volume = self.read_series(self.series_dirs[i], self.shapes[i])
                volume = crop_volume(volume, self.reverses[i], self.centroids[i], self.x_spans[i], self.y_spans[i])
                volume = self.standardize_volume(volume)
                if self.n_channels > 1:
                    volume = np.repeat(volume, self.n_channels, -1)
                volumes.append(volume)
                # print(self.series_dirs[i], volume.shape)
                del volume
            volumes = np.array(volumes, dtype=np.float64)

            return volumes, y

    def on_epoch_end(self) -> None:
        """Updates indexes after each epoch"""
        with device(dev):
            self.indexes = np.arange(len(self.series_dirs))
            # if self.shuffle == True:
            #     np.random.shuffle(self.indexes, seed=1)
            print('Epoch end')

    def n_samples(self) -> int:
        """Get the number of image samples"""
        return len(self.series_dirs)

    def _get_prefix(self) -> str:
            return prefixNSCLC

    def _get_valid_pids(self, label_filename: str) -> Set[str]:
        """Get valid PIDs with available genomic data"""
        f = open(label_filename, 'r')
        reader = DictReader(f, delimiter='\t')
        pids = set()
        for row in reader:
            pids.add(row['PID'])
        f.close()
        return pids

    def _get_series_metadata(self, metadata_field_name: str, directory_field_name: str, pid_field_name: str) \
            -> Tuple[
                List[str], # series_dirs
                List[str], # pids
                List[Tuple[int, int]], # shapes
                List[Tuple[int, int, int]], # centroids
                List[Tuple[int, int]], # x_spans
                List[Tuple[int, int]], # y_spans
                List[bool] # reverses
            ]:
        """Get metadata of series (directories, patient ids, image shapes)"""
        f = open(metadata_field_name, 'r')
        reader = DictReader(f, delimiter=',')
        dirs = []
        pids = []
        shapes = []
        centroids = []
        x_spans = []
        y_spans = []
        reverses = []
        for row in reader:
            if row[pid_field_name] in self.valid_pids:  # TODO add other checks/filters
                dirs.append(row[directory_field_name][2:] + '/')
                pids.append(row[pid_field_name])
                shapes.append((int(row['Slice Rows']), int(row['Slice Columns'])))
                centroids.append(
                    (
                        int(row['OriginZ']) + int(row['SegOffsetZ']),
                        int(row['OriginX']),
                        int(row['OriginY'])
                    )
                )
                x_spans.append(
                    (
                        int(row['OriginX']) - (int(row['SegSpanX']) // 2),
                        int(row['OriginX']) + (int(row['SegSpanX']) // 2)
                    )
                )
                y_spans.append(
                    (
                        int(row['OriginY']) - (int(row['SegSpanY']) // 2),
                        int(row['OriginY']) + (int(row['SegSpanY']) // 2)
                    )
                )
                reverses.append(bool(int(row['SegReverse'])))

        f.close()
        # return dirs[:32], pids[:32], shapes[:32], centroids[:32], x_spans[:32], y_spans[:32], reverses[:32]
        return dirs, pids, shapes, centroids, x_spans, y_spans, reverses

    def _get_target_labels(self, label_filename: str, index_range: Tuple[int, int], dtype: type) \
            -> Tuple[np.array, List[str]]:
        """Get target and target names"""
        f = open(label_filename, 'r')
        genomic_values = list(np.genfromtxt(f, dtype=dtype, delimiter='\t', skip_header=1,
                                            usecols=range(index_range[0], index_range[1])))
        f.close()
        f = open(label_filename, 'r')
        genomic_pid_labels = np.genfromtxt(f, dtype=str, delimiter='\t', skip_header=1, usecols=(0,))
        genomic_pid_labels = {pid: i for i, pid in enumerate(list(genomic_pid_labels))}
        f.close()
        f = open(label_filename, 'r')
        target_names = list(DictReader(f, delimiter='\t').fieldnames)[index_range[0]:index_range[1]]
        f.close()
        # print((len(genomic_values), len(genomic_values[0])), len(genomic_pid_labels))
        target_list = []
        for p in self.pids:
            target_list.append(genomic_values[genomic_pid_labels[p]])
        target = np.array(target_list)
        if target.ndim == 1:
            # for i in range(target.shape[0]):
            #     if target[i] == 0:
            #         target[i] = -1
            target = np.reshape(target, (-1, 1))
        # print(target.shape, target.dtype)
        # return target[:32, :], target_names
        return target, target_names

    def _scale_target_labels(self) -> None:
        # print(np.histogram(self.targetLabels, bins=[0, 1, 5, 10, 25, 50, 100, 1000, 100000000]))
        for i in range(self.targetLabels.shape[0]):
            for j in range(self.targetLabels.shape[1]):
                if self.targetLabels[i, j] > 25:
                    self.targetLabels[i, j] = 25
        # print(np.histogram(self.targetLabels, bins=[0, 1, 5, 10, 25, 50, 100, 1000, 100000000]))
        ss_transformed_target = StandardScaler().fit_transform(self.targetLabels)
        # print(np.histogram(ss_transformed_target, bins=10))
        mms_transformed_target = MinMaxScaler().fit_transform(ss_transformed_target)
        # print(np.histogram(mms_transformed_target, bins=10))
        self.targetLabels = mms_transformed_target

    def get_preferred_shape(self, index: int) -> None:
        """Get preferred shape of series"""
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        dirs = [self.series_dirs[k] for k in indexes]
        for path in dirs:
            shape_counts = dict()
            prefix = self._get_prefix()
            for fName in listdir(prefix + path):
                if isfile(prefix + path + fName) and fName[-4:] == '.dcm':
                    ds = dcmread(prefix + path + fName)
                    current_shape = (ds.Rows, ds.Columns)
                    if current_shape in shape_counts:
                        shape_counts[current_shape] = shape_counts[current_shape] + 1
                    else:
                        shape_counts[current_shape] = 1
            preferred_shape = max(shape_counts, key=shape_counts.get)
            print(path, preferred_shape)
        return

    def read_series(self, path: str, shape: Tuple[int, int]) -> np.array:
        """Read one volume"""
        slices = []
        prefix = self._get_prefix()
        for fName in listdir(prefix + path):
            if isfile(prefix + path + fName) and fName[-4:] == '.dcm':
                ds = dcmread(prefix + path + fName)

                if (ds.Rows, ds.Columns) == shape:
                    slices.append(ds.pixel_array.astype("int32"))
        volume = np.stack(slices, axis=0)
        return volume

    def standardize_volume(self, volume: np.array) -> np.array:
        """Standardize volume (normalize, rotate, resize)"""
        volume = volume.astype('float32')
        # Histogram equalization
        volume = equalize_hist(volume, nbins=256)
        # Scale
        volume = zoom_volume_2d(volume, dim=self.dim)
        # Shift
        if self.shift_factor != 0:
            volume = shift(volume, (0, self.shift_factor))
        # Rotate
        if self.rotation_angle != 0:
            volume = rotate(volume, angle=self.rotation_angle, reshape=False)
        volume = expand_dims(volume, axis=2)
        return volume
