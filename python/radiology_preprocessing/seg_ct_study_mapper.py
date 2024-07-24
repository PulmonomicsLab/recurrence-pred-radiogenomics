from csv import DictReader

f_seg = open('/home/sudipto/Templates/manifest-1695119508564/metadata.csv', 'r')
seg_reader = DictReader(f_seg, delimiter=',')
seg_map = dict()
for row in seg_reader:
    seg_map[row['Study UID']] = row['File Location']

print(len(seg_map))

f_ct = open('input/Radiology/image-labels3.csv', 'r')
ct_reader = DictReader(f_ct, delimiter=',')
for row in ct_reader:
    study_id = row['Study UID']
    if study_id in seg_map:
        print(study_id + ',' + seg_map[study_id])
    else:
        print(study_id + ',')
