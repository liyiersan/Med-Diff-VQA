import os
import cv2
import numpy as np
from PIL import Image, ImageFile
import pickle
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

ImageFile.LOAD_TRUNCATED_IMAGES = True

def resize(array, size, keep_ratio=False, resample=Image.LANCZOS):
    im = Image.fromarray(array)

    if keep_ratio:
        im.thumbnail((size, size), resample)
    else:
        im = im.resize((size, size), resample)

    return im

def process_image(file, s_path, out_path, size, n):
    if file == 'index.html' or file.endswith('.gstmp'):
        return None, None
    new_filename = os.path.join(out_path, file.replace('.jpg', '.png'))
    record = {}
    file_path = os.path.join(s_path, file)
    im = Image.open(file_path)
    record['image'] = file.replace('.jpg', '')
    record['height'] = im.size[0]
    record['width'] = im.size[1]
    if not os.path.exists(new_filename):
        im = im.resize((size, size), Image.LANCZOS)
        im.save(new_filename)
    n += 1
    return record, n

def mimic_jpg2png(data_path, out_path, max_workers):
    data_path = os.path.join(data_path, 'files')
    p_folder = os.listdir(data_path)
    p_folder = [x for x in p_folder if not x.endswith('.zip')]
    size = 1024
    n = 0
    record_list = []
    mimic_shapeid = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for p_fold in p_folder:
            if p_fold == 'index.html':
                continue
            p_path = os.path.join(data_path, p_fold)
            pp_folder = os.listdir(p_path)
            for pp_fold in pp_folder:
                if pp_fold == 'index.html':
                    continue
                pp_path = os.path.join(p_path, pp_fold)
                if not os.path.isdir(pp_path):
                    continue
                s_folder = os.listdir(pp_path)
                for s_fold in s_folder:
                    if s_fold == 'index.html':
                        continue
                    s_path = os.path.join(pp_path, s_fold)
                    if not os.path.isdir(s_path):
                        continue
                    files = os.listdir(s_path)
                    for file in files:
                        futures.append(executor.submit(process_image, file, s_path, out_path, size, n))

        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            if result[0] is not None:
                record, image_id = result
                record_list.append(record)
                mimic_shapeid[record['image']] = image_id
                n = image_id  # update the image id

    if not os.path.exists('pkls'):
        os.mkdir('pkls')
    with open('pkls/mimic_shape_full.pkl', 'wb') as f:
        pickle.dump(record_list, f)
        print('file saved')
    with open('pkls/mimic_shapeid_full.pkl', 'wb') as f:
        pickle.dump(mimic_shapeid, f)
        print('file saved')

    dicom2id = {record['image']: i for i, record in enumerate(record_list)}
    with open('pkls/dicom2id.pkl', "wb") as tf:
        pickle.dump(dicom2id, tf)
        print('dicom2id saved')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--mimic_path", type=str, default='/data/mimic-cxr-jpg-2.1.0/', required=False, help="path to mimic-cxr-jpg dataset")
    parser.add_argument("-o", "--out_path", type=str, default='/data/mimic-cxr-jpg-2.1.0/mimic_cxr_png', required=False, help="path to output png dataset")
    parser.add_argument("-w", "--workers", type=int, default=os.cpu_count(), required=False, help="number of worker threads")
    args = parser.parse_args()
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path, exist_ok=True)
    mimic_jpg2png(data_path=args.mimic_path, out_path=args.out_path, max_workers=args.workers)

if __name__ == '__main__':
    main()
