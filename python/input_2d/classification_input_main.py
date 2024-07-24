from .data_generator import DataGeneratorSegmentation2D
from .tfrecord import create_tf_record_from_generator
from .tfrecord import parse_example_classification, parse_matrices_classification
from .tfrecord import check_equality_between_gen_and_tfrd
from .tfrecord import get_dataset
from os.path import isfile
from os import listdir

# CV + blind
# TFRPrefixPath = '/home/sudipto/Templates/TFR_classification/TFR_all/'
# dg_r0_s0_3c = DataGeneratorSegmentation2D('input/Radiology/seg-labels2.csv',
#                                           'input/Genomic/genomic_smote_Boruta_SVM_140.tsv',
#                                           target_index_range=(1, 2), n_channels=3,
#                                           rotation_angle=0, shift_factor=0)
# dg_r10p_s0_3c = DataGeneratorSegmentation2D('input/Radiology/seg-labels2.csv',
#                                             'input/Genomic/genomic_smote_Boruta_SVM_140.tsv',
#                                             target_index_range=(1, 2), n_channels=3,
#                                             rotation_angle=10, shift_factor=0)
# dg_r10n_s0_3c = DataGeneratorSegmentation2D('input/Radiology/seg-labels2.csv',
#                                             'input/Genomic/genomic_smote_Boruta_SVM_140.tsv',
#                                             target_index_range=(1, 2), n_channels=3,
#                                             rotation_angle=-10, shift_factor=0)
# dg_r20p_s0_3c = DataGeneratorSegmentation2D('input/Radiology/seg-labels2.csv',
#                                             'input/Genomic/genomic_smote_Boruta_SVM_140.tsv',
#                                             target_index_range=(1, 2), n_channels=3,
#                                             rotation_angle=20, shift_factor=0)
# dg_r20n_s0_3c = DataGeneratorSegmentation2D('input/Radiology/seg-labels2.csv',
#                                             'input/Genomic/genomic_smote_Boruta_SVM_140.tsv',
#                                             target_index_range=(1, 2), n_channels=3,
#                                             rotation_angle=-20, shift_factor=0)
# dg_r0_s20p_3c = DataGeneratorSegmentation2D('input/Radiology/seg-labels2.csv',
#                                             'input/Genomic/genomic_smote_Boruta_SVM_140.tsv',
#                                             target_index_range=(1, 2), n_channels=3,
#                                             rotation_angle=0, shift_factor=20)
# dg_r0_s20n_3c = DataGeneratorSegmentation2D('input/Radiology/seg-labels2.csv',
#                                             'input/Genomic/genomic_smote_Boruta_SVM_140.tsv',
#                                             target_index_range=(1, 2), n_channels=3,
#                                             rotation_angle=0, shift_factor=-20)
# create_tf_record_from_generator(dg_r0_s0_3c, TFRPrefixPath, 16, data_type='classification')
# create_tf_record_from_generator(dg_r10p_s0_3c, TFRPrefixPath, 16, data_type='classification')
# create_tf_record_from_generator(dg_r10n_s0_3c, TFRPrefixPath, 16, data_type='classification')
# create_tf_record_from_generator(dg_r20p_s0_3c, TFRPrefixPath, 16, data_type='classification')
# create_tf_record_from_generator(dg_r20n_s0_3c, TFRPrefixPath, 16, data_type='classification')
# create_tf_record_from_generator(dg_r0_s20p_3c, TFRPrefixPath, 16, data_type='classification')
# create_tf_record_from_generator(dg_r0_s20n_3c, TFRPrefixPath, 16, data_type='classification')

# CV
# TFRPrefixPath = '/home/sudipto/Templates/TFR_classification/TFR_CV/'
# dg_r0_s0_3c = DataGeneratorSegmentation2D('input/Radiology/seg-labels3.csv',
#                                           'input/Genomic/genomic_smote_Boruta_SVM_140.tsv',
#                                           target_index_range=(1, 2), n_channels=3,
#                                           rotation_angle=0, shift_factor=0)
# dg_r10p_s0_3c = DataGeneratorSegmentation2D('input/Radiology/seg-labels3.csv',
#                                             'input/Genomic/genomic_smote_Boruta_SVM_140.tsv',
#                                             target_index_range=(1, 2), n_channels=3,
#                                             rotation_angle=10, shift_factor=0)
# dg_r10n_s0_3c = DataGeneratorSegmentation2D('input/Radiology/seg-labels3.csv',
#                                             'input/Genomic/genomic_smote_Boruta_SVM_140.tsv',
#                                             target_index_range=(1, 2), n_channels=3,
#                                             rotation_angle=-10, shift_factor=0)
# dg_r20p_s0_3c = DataGeneratorSegmentation2D('input/Radiology/seg-labels3.csv',
#                                             'input/Genomic/genomic_smote_Boruta_SVM_140.tsv',
#                                             target_index_range=(1, 2), n_channels=3,
#                                             rotation_angle=20, shift_factor=0)
# dg_r20n_s0_3c = DataGeneratorSegmentation2D('input/Radiology/seg-labels3.csv',
#                                             'input/Genomic/genomic_smote_Boruta_SVM_140.tsv',
#                                             target_index_range=(1, 2), n_channels=3,
#                                             rotation_angle=-20, shift_factor=0)
# dg_r0_s20p_3c = DataGeneratorSegmentation2D('input/Radiology/seg-labels3.csv',
#                                             'input/Genomic/genomic_smote_Boruta_SVM_140.tsv',
#                                             target_index_range=(1, 2), n_channels=3,
#                                             rotation_angle=0, shift_factor=20)
# dg_r0_s20n_3c = DataGeneratorSegmentation2D('input/Radiology/seg-labels3.csv',
#                                             'input/Genomic/genomic_smote_Boruta_SVM_140.tsv',
#                                             target_index_range=(1, 2), n_channels=3,
#                                             rotation_angle=0, shift_factor=-20)
# create_tf_record_from_generator(dg_r0_s0_3c, TFRPrefixPath, 16, data_type='classification')
# create_tf_record_from_generator(dg_r10p_s0_3c, TFRPrefixPath, 16, data_type='classification')
# create_tf_record_from_generator(dg_r10n_s0_3c, TFRPrefixPath, 16, data_type='classification')
# create_tf_record_from_generator(dg_r20p_s0_3c, TFRPrefixPath, 16, data_type='classification')
# create_tf_record_from_generator(dg_r20n_s0_3c, TFRPrefixPath, 16, data_type='classification')
# create_tf_record_from_generator(dg_r0_s20p_3c, TFRPrefixPath, 16, data_type='classification')
# create_tf_record_from_generator(dg_r0_s20n_3c, TFRPrefixPath, 16, data_type='classification')

# Blind
# TFRPrefixPath = '/home/sudipto/Templates/TFR_classification/TFR_blind/'
# dg_r0_s0_3c = DataGeneratorSegmentation2D('input/Radiology/seg-labels4.csv',
#                                           'input/Genomic/genomic_smote_Boruta_SVM_140.tsv',
#                                           target_index_range=(1, 2), n_channels=3,
#                                           rotation_angle=0, shift_factor=0)
# dg_r10p_s0_3c = DataGeneratorSegmentation2D('input/Radiology/seg-labels4.csv',
#                                             'input/Genomic/genomic_smote_Boruta_SVM_140.tsv',
#                                             target_index_range=(1, 2), n_channels=3,
#                                             rotation_angle=10, shift_factor=0)
# dg_r10n_s0_3c = DataGeneratorSegmentation2D('input/Radiology/seg-labels4.csv',
#                                             'input/Genomic/genomic_smote_Boruta_SVM_140.tsv',
#                                             target_index_range=(1, 2), n_channels=3,
#                                             rotation_angle=-10, shift_factor=0)
# dg_r20p_s0_3c = DataGeneratorSegmentation2D('input/Radiology/seg-labels4.csv',
#                                             'input/Genomic/genomic_smote_Boruta_SVM_140.tsv',
#                                             target_index_range=(1, 2), n_channels=3,
#                                             rotation_angle=20, shift_factor=0)
# dg_r20n_s0_3c = DataGeneratorSegmentation2D('input/Radiology/seg-labels4.csv',
#                                             'input/Genomic/genomic_smote_Boruta_SVM_140.tsv',
#                                             target_index_range=(1, 2), n_channels=3,
#                                             rotation_angle=-20, shift_factor=0)
# dg_r0_s20p_3c = DataGeneratorSegmentation2D('input/Radiology/seg-labels4.csv',
#                                             'input/Genomic/genomic_smote_Boruta_SVM_140.tsv',
#                                             target_index_range=(1, 2), n_channels=3,
#                                             rotation_angle=0, shift_factor=20)
# dg_r0_s20n_3c = DataGeneratorSegmentation2D('input/Radiology/seg-labels4.csv',
#                                             'input/Genomic/genomic_smote_Boruta_SVM_140.tsv',
#                                             target_index_range=(1, 2), n_channels=3,
#                                             rotation_angle=0, shift_factor=-20)
# create_tf_record_from_generator(dg_r0_s0_3c, TFRPrefixPath, 16, data_type='classification')
# create_tf_record_from_generator(dg_r10p_s0_3c, TFRPrefixPath, 16, data_type='classification')
# create_tf_record_from_generator(dg_r10n_s0_3c, TFRPrefixPath, 16, data_type='classification')
# create_tf_record_from_generator(dg_r20p_s0_3c, TFRPrefixPath, 16, data_type='classification')
# create_tf_record_from_generator(dg_r20n_s0_3c, TFRPrefixPath, 16, data_type='classification')
# create_tf_record_from_generator(dg_r0_s20p_3c, TFRPrefixPath, 16, data_type='classification')
# create_tf_record_from_generator(dg_r0_s20n_3c, TFRPrefixPath, 16, data_type='classification')

# fileNames = sorted(
#     [TFRPrefixPath + f for f in listdir(TFRPrefixPath) if isfile(TFRPrefixPath+f) and f[-9:] == '.tfrecord']
# )
# k = 0
# print(fileNames)
# dataset = get_dataset(fileNames, batch_size=1, shuffle=False, deterministic=True, parser=parse_example_classification)
# for s in dataset.take(-1).as_numpy_iterator():
#     print(k, '->', s['volume'].shape, s['y'].shape)
#     k += 1
# check_equality_between_gen_and_tfrd(
#     [dg_r10n_s0_3c, dg_r20n_s0_3c, dg_r0_s20n_3c, dg_r0_s0_3c, dg_r0_s20p_3c, dg_r10p_s0_3c, dg_r20p_s0_3c],
#     dataset
# )
