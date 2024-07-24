from typing import List, Tuple
from math import ceil
import numpy as np
import tensorflow as tf
from python.input_2d.data_generator import DataGeneratorSegmentation2D


feature_descriptor_classification = {
    'volume': tf.io.FixedLenFeature((128, 128, 3), tf.float32),
    'y': tf.io.FixedLenFeature((1,), tf.int64),
}

feature_descriptor_combined = {   # 3-channel DHW
    'volume': tf.io.FixedLenFeature((64, 128, 128, 3), tf.float32),
    'genomic': tf.io.FixedLenFeature((140,), tf.float32),
    'y': tf.io.FixedLenFeature((1,), tf.int64),
}

feature_descriptor_intermediate_classification = {   # 3-channel DHW
    'volume': tf.io.FixedLenFeature((512,), tf.float32),
    'y': tf.io.FixedLenFeature((1,), tf.int64),
}

feature_descriptor_intermediate_combined = {   # 3-channel DHW
    'volume': tf.io.FixedLenFeature((652,), tf.float32),
    'y': tf.io.FixedLenFeature((1,), tf.int64),
}


def parse_example_classification(inp):
    parsed_features = tf.io.parse_single_example(inp, feature_descriptor_classification)
    return parsed_features


def parse_matrices_classification(inp):
    parsed_features = tf.io.parse_single_example(inp, feature_descriptor_classification)
    return parsed_features['volume'], parsed_features['y']


def parse_example_combined(inp):
    parsed_features = tf.io.parse_single_example(inp, feature_descriptor_combined)
    return parsed_features


def parse_matrices_combined(inp):
    parsed_features = tf.io.parse_single_example(inp, feature_descriptor_combined)
    return [parsed_features['volume'], parsed_features['genomic']], parsed_features['y']


def parse_example_intermediate_classification(inp):
    parsed_features = tf.io.parse_single_example(inp, feature_descriptor_intermediate_classification)
    return parsed_features


def parse_matrices_intermediate_classification(inp):
    parsed_features = tf.io.parse_single_example(inp, feature_descriptor_intermediate_classification)
    return parsed_features['volume'], parsed_features['y']


def parse_example_intermediate_combined(inp):
    parsed_features = tf.io.parse_single_example(inp, feature_descriptor_intermediate_combined)
    return parsed_features


def parse_matrices_intermediate_combined(inp):
    parsed_features = tf.io.parse_single_example(inp, feature_descriptor_intermediate_combined)
    return parsed_features['volume'], parsed_features['y']


def create_tf_record_from_numpy(data: np.array, target: np.array, tfr_prefix_path: str, samples_per_file: int,
                                data_type: str = 'classification') -> None:
    n_samples = data.shape[0]
    n_files = ceil(n_samples / samples_per_file)
    sample_counter = 0
    for nf in range(n_files):
        print('----------------------------- File ' + str(nf) + ' -----------------------------')
        writer = tf.io.TFRecordWriter(tfr_prefix_path + str(nf) + '.tfrecord')
        print(tfr_prefix_path + str(nf) + '.tfrecord')
        end = sample_counter + samples_per_file
        if end > n_samples:
            end = n_samples
        for i in range(sample_counter, end):
            if data_type == 'classification' or data_type == 'combined':
                x, y = data[i, :].reshape(1, -1), np.array([target[i]]).reshape(1, -1)
                print(i, nf, '->', x.shape, y.shape, y)
                feature = {
                    'volume': tf.train.Feature(float_list=tf.train.FloatList(value=x[0].flatten())),
                    'y': tf.train.Feature(int64_list=tf.train.Int64List(value=y[0])),
                }
            else:
                raise ValueError(
                    'Error !!! Invalid \'data_type\' argument. Must be {regression, classification, combined} ...')
            example = tf.train.Example(features=tf.train.Features(feature=feature))  # feature
            writer.write(example.SerializeToString())
        sample_counter = end
        writer.close()


def create_tf_record_from_generator(gen: DataGeneratorSegmentation2D, tfr_prefix_path: str, samples_per_file: int,
                                    data_type: str = 'classification') -> None:
    n_files = ceil(gen.n_samples() / samples_per_file)
    sample_counter = 0
    for nf in range(n_files):
        print('----------------------------- File ' + str(nf) + ' -----------------------------')
        writer = tf.io.TFRecordWriter(
            tfr_prefix_path +
            'tfr_r' + str(gen.rotation_angle) +
            '_s' + str(gen.shift_factor) +
            '_' + str(nf) + '.tfrecord'
        )
        print(tfr_prefix_path +
              'tfr_r' + str(gen.rotation_angle) +
              '_s' + str(gen.shift_factor) +
              '_' + str(nf) + '.tfrecord')
        end = sample_counter + samples_per_file
        if end > gen.n_samples():
            end = gen.n_samples()
        for i in range(sample_counter, end):
            if data_type == 'classification':
                x, y = gen.__getitem__(i)
                print(i, nf, '->', x[0].shape, y.shape)
                feature = {
                    'volume': tf.train.Feature(float_list=tf.train.FloatList(value=x[0].flatten())),
                    'y': tf.train.Feature(int64_list=tf.train.Int64List(value=y[0])),
                }
            elif data_type == 'combined':
                x, genomic, y = gen.__getitem__(i)
                print(i, nf, '->', x[0].shape, genomic.shape, y.shape)
                feature = {
                    'volume': tf.train.Feature(float_list=tf.train.FloatList(value=x[0].flatten())),
                    'genomic': tf.train.Feature(float_list=tf.train.FloatList(value=genomic[0])),
                    'y': tf.train.Feature(int64_list=tf.train.Int64List(value=y[0])),
                }
            else:
                raise ValueError(
                    'Error !!! Invalid \'data_type\' argument. Must be {regression, classification, combined} ...')
            example = tf.train.Example(features=tf.train.Features(feature=feature))  # feature
            writer.write(example.SerializeToString())
        sample_counter = end
        writer.close()


def check_equality_between_gen_and_tfrd(gen: List[DataGeneratorSegmentation2D],
                                        dataset: tf.data.TFRecordDataset) -> None:
    samples = dataset.take(-1).as_numpy_iterator()
    # print(samples)
    k1 = 0
    k2 = 0
    for sample in samples:
        if gen[k2].data_type == 'combined':
            x1, genomic1, y1 = sample['volume'], sample['genomic'], sample['y']
            x, genomic, y = gen[k2].__getitem__(k1)
            print('r', str(gen[k2].rotation_angle), 's', str(gen[k2].shift_factor), k2, k1, '->',
                  x1.shape, genomic1.shape, y1.shape, x.shape, genomic.shape, y.shape)
            x_eq, genomic_eq, y_eq = np.array_equal(x, x1), np.array_equal(genomic, genomic1), np.array_equal(y, y1)
            print(k1, '->', x_eq, genomic_eq, y_eq)
            del genomic1
            del genomic
        else:
            x1, y1 = sample['volume'], sample['y']
            x, y = gen[k2].__getitem__(k1)
            print('r', str(gen[k2].rotation_angle), 's', str(gen[k2].shift_factor), k2, k1, '->',
                  x1.shape, y1.shape, x.shape, y.shape)
            x_eq, y_eq = np.array_equal(x, x1), np.array_equal(y, y1)
            print(k1, '->', x_eq, y_eq)
        k1 += 1
        if k1 == gen[k2].n_samples():
            k1 = 0
            k2 += 1
        del x1
        del y1
        del x
        del y
    print()


def check_equality_between_numpy_and_tfrd(data: np.array, target: np.array, dataset: tf.data.TFRecordDataset) -> None:
    samples = dataset.take(-1).as_numpy_iterator()
    # print(samples)
    k = 0
    for sample in samples:
        x1, y1 = sample['volume'], sample['y']
        x, y = data[k, :], target[k]
        # x, y = gen[k2].__getitem__(k1)
        print(k, '->', x1.shape, y1.shape, x.shape, y.shape)
        x_eq, y_eq = np.array_equal(x, x1), np.array_equal(y, y1)
        print(k, '->', x_eq, y_eq)
        k += 1
        del x1
        del y1
        del x
        del y
    print()


def get_dataset(filenames: List[str], batch_size: int, shuffle: bool = True, buffer_size: int = None,
                deterministic: bool = False, parser=parse_matrices_classification) -> tf.data.TFRecordDataset:
    dataset = (
        tf.data.TFRecordDataset(filenames)
        .map(parser, num_parallel_calls=tf.data.AUTOTUNE, deterministic=deterministic)
        .batch(batch_size)
    )
    if shuffle:
        if buffer_size is None:
            buffer_size = batch_size * 10
        dataset = dataset.shuffle(buffer_size=buffer_size, seed=0, reshuffle_each_iteration=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def check_dataset_equality(dataset1, dataset2) -> None:
    it1 = dataset1.take(-1).as_numpy_iterator()
    it2 = dataset2.take(-1).as_numpy_iterator()
    k = 0
    for sample1 in it1:
        sample2 = next(it2)
        x_eq, y_eq = np.array_equal(sample1['volume'], sample2['volume']), np.array_equal(sample1['y'], sample2['y'])
        print(k, sample1['volume'].shape, sample2['volume'].shape, sample1['y'].shape, sample2['y'].shape, x_eq, y_eq)
        k += 1
    print()


def verbose_dataset(dataset: tf.data.TFRecordDataset, x_label, y_label) -> None:
    k = 0
    for s in dataset.take(-1).as_numpy_iterator():
        print(k, '->', s[x_label].shape, s[y_label].shape)
        k += 1
    print()
