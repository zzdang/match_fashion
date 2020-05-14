import face_model
import torch
from k_reciprocal import KReciprocal
import argparse
import json
import base64
import numpy as np
import cv2,os,json
from tqdm import tqdm

parser = argparse.ArgumentParser(description='do verification')
# general
parser.add_argument('--img_dir', help='test config file path')
parser.add_argument('--det_config', help='test config file path')
parser.add_argument('--det_checkpoint', help='checkpoint file')
parser.add_argument('--batch_size', help='batch size for recognization')
parser.add_argument('--det_device', type=int, default=0, help='CUDA device id')
parser.add_argument('--score_thr', type=float, default=0.3, help='bbox score threshold')
parser.add_argument('--image-size', default='224,224', help='')
parser.add_argument('--model', default='../model/softmax,50', help='path to load model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
args = parser.parse_args()

model = face_model.FaceModel(args)
im_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
mp4_extensions = ['.mp4', '.MP4']


def image_resize(img):
  h,w,c = img.shape
  img_dst = np.ones((224, 224, 3), dtype=np.uint8) * 128
  if h >= w:
    dst_h = 224;
    dst_w = int(w / h * 224.0)
  else:
    dst_w = 224;
    dst_h = int(h / w * 224.0)
  img = cv2.resize(img, (dst_w, dst_h))
  img_dst[int((224 - dst_h) / 2):int((224 - dst_h) / 2) + dst_h, int((224 - dst_w) / 2):int((224 - dst_w) / 2) + dst_w, :] = img
  img_dst = img_dst.transpose((2,0,1))
  return img_dst

def get_image(path):
  if len(path) == 1:
    url = path[0]
    image = cv2.imread(url, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image_resize(image)
  else:
    image = []
    for p in path:
      _image = cv2.imread(p, cv2.IMREAD_COLOR)
      _image = cv2.cvtColor(_image, cv2.COLOR_BGR2RGB)
      _image = image_resize(_image)
      image.append(_image)

  return image


def do_flip(data):
  for idx in range(data.shape[0]):
    data[idx,:,:] = np.fliplr(data[idx,:,:])

def get_all_feature(ids_path=None, batch_size = 128):
  target_feature, ids = [], []
  target_images_all = [[], []]
  idx =  0
  for root, dir_names, file_names in os.walk(ids_path):
    for file_name in file_names:
      idx += 1
      print(idx/100000)
      if os.path.splitext(file_name)[1] not in im_extensions:
        continue
      img_path = os.path.join(root, file_name)
      img = cv2.imread(img_path)
      target_images, bboxes = model.get_all_faces(img, max_len=1333, score_thresh=0.3)

      for target_img, bbox in zip(target_images, bboxes):   #### can be use N batch
        target_images_all[0].append(target_img)
        _img = np.copy(target_img)
        do_flip(_img)
        target_images_all[1].append(_img)
        # _feature = model.get_feature(target_img, True)
        # target_feature.append(_feature)
        ids.append([root.split('/')[-1], file_name.split('.')[0], bbox[:4]])
        if len(target_images_all[0]) == batch_size:
          target_feature.extend(model.get_features(target_images_all, batch_size))
          target_images_all = [[], []]
  if len(target_images_all[0]) > 0:
    target_feature.extend(model.get_features(target_images_all, len(target_images_all[0])))
  return np.array(target_feature), ids

def det_match_v(vedio_path=None, gallery_fea=None, ids=None, batch_size=128):
  res_dict = {}
  mp4_ids, frame_indexs, frame_boxes = [], [], []
  query_fea = []
  hps = {
    "k1": 20,
    "k2": 6,
    "lambda_value": 0.3,
  }
  re_rank = KReciprocal(hps)
  for root, dir_names, file_names in os.walk(vedio_path):
    for file_name in tqdm(file_names):
      if os.path.splitext(file_name)[1] not in mp4_extensions:
        continue

      mp4_path = os.path.join(root, file_name)
      cap = cv2.VideoCapture(mp4_path)
      num_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
      frame_range = 40
      mp4_id = file_name.split('.')[0]
      source_image_all = [[], []]
      for frame_idx in range(int(num_frame//frame_range)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx*frame_range)
        _, img = cap.read()
        source_images, bboxes = model.get_all_faces(img, max_len=1333, score_thresh=0.5)
        for source_image, bbox in zip(source_images, bboxes):
          source_image_all[0].append(source_image)
          _img = np.copy(source_image)
          do_flip(_img)
          source_image_all[1].append(_img)
          frame_index, frame_box = int(frame_idx * frame_range), bbox[:4]
          mp4_ids.append(mp4_id)
          frame_indexs.append(frame_index)
          frame_boxes.append(frame_box)

      if len(source_image_all[0]) < batch_size:
        batch_size = len(source_image_all[0])
      query_fea.extend( model.get_features(source_image_all, batch_size ) )

  query_fea, gallery_fea = torch.tensor(query_fea), torch.tensor(gallery_fea)
  l2_dis = re_rank._cal_dis(query_fea, gallery_fea)
  final_dist, sort_idx = re_rank(query_fea, gallery_fea, l2_dis)
  mp4_ids, frame_indexs, frame_boxes = np.array(mp4_ids), np.array(frame_indexs), np.array(frame_boxes)

  for id in range(len(file_names)):
    mp4_id = file_names[id].split('.')[0]
    mp4_idx = mp4_ids == mp4_id
    print(mp4_id)
    item_id, frame_index, img_name, item_box, frame_box = model.sims(final_dist[mp4_idx], l2_dis.numpy()[mp4_idx], \
                          frame_indexs[mp4_idx], frame_boxes[mp4_idx], ids)
    print(item_id, frame_index, img_name)
    if len(item_id) == 0:
      continue

    res_dict[mp4_id] = {}
    res_dict[mp4_id]['item_id'] = item_id
    res_dict[mp4_id]['frame_index'] = frame_index.tolist()
    res_dict[mp4_id]['result'] = []
    for i in range(len(img_name)):
      res_dict[mp4_id]['result'].append( {'img_name':img_name[i], \
                                'item_box':item_box[i].tolist(), 'frame_box':frame_box[i].tolist()} )

  return res_dict


if __name__ == '__main__':

  img_dir = args.img_dir + '/image/'
  video_dir = args.img_dir + '/video/'
  gallery_fea, ids = get_all_feature(img_dir, batch_size=64)
  res_dict = det_match_v(video_dir, gallery_fea, ids, batch_size=64)

  with open('result.json', 'w', encoding='utf-8') as json_file:
    json.dump(res_dict, json_file, indent=4, ensure_ascii=False)
