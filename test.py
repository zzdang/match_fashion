import face_model
import argparse
import json
import base64
import numpy as np
import cv2,os


parser = argparse.ArgumentParser(description='do verification')
# general
parser.add_argument('--det_config', help='test config file path')
parser.add_argument('--det_checkpoint', help='checkpoint file')
parser.add_argument('--det_device', type=int, default=0, help='CUDA device id')
parser.add_argument('--score_thr', type=float, default=0.3, help='bbox score threshold')
parser.add_argument('--image-size', default='224,224', help='')
parser.add_argument('--model', default='../model/softmax,50', help='path to load model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
args = parser.parse_args()

model = face_model.FaceModel(args)
im_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
target_feature = []
ids = []
global im_extensions
global target_feature
global ids

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

def ver():
  try:
    d_p = ['/media/chen/6f586f18-792a-40fd-ada6-59702fb5dabc/comp/insightface/out/0_109090/1.jpg',
           '/media/chen/6f586f18-792a-40fd-ada6-59702fb5dabc/comp/insightface/out/0_109118/1.jpg',
           '/media/chen/6f586f18-792a-40fd-ada6-59702fb5dabc/comp/insightface/out/0_109122/3.jpg',
           '/media/chen/6f586f18-792a-40fd-ada6-59702fb5dabc/comp/insightface/out/0_109094/1.jpg']
    s_p = ['/media/chen/6f586f18-792a-40fd-ada6-59702fb5dabc/comp/insightface/out/0_109090/3.jpg']
    d_p = ['/media/chen/6f586f18-792a-40fd-ada6-59702fb5dabc/comp/insightface/out/0_109168/0.jpg', '/media/chen/6f586f18-792a-40fd-ada6-59702fb5dabc/comp/insightface/out/0_109168/1.jpg',
           '/media/chen/6f586f18-792a-40fd-ada6-59702fb5dabc/comp/insightface/out/0_109168/2.jpg', '/media/chen/6f586f18-792a-40fd-ada6-59702fb5dabc/comp/insightface/out/0_109168/3.jpg',
           '/media/chen/6f586f18-792a-40fd-ada6-59702fb5dabc/comp/insightface/out/0_109168/4.jpg', '/media/chen/6f586f18-792a-40fd-ada6-59702fb5dabc/comp/insightface/out/0_109168/5.jpg']
    s_p =['/media/chen/6f586f18-792a-40fd-ada6-59702fb5dabc/comp/insightface/out/0_109168/5.jpg']
    source_image = get_image(s_p)
    if source_image is None:
      print('source image is None')
      return '-1'
    assert not isinstance(source_image, list)
    print(source_image.shape)
    target_image = get_image(d_p)
    if target_image is None:
      print('target image is None')
      return '-1'
    #print(target_image.shape)
    if not isinstance(target_image, list):
      target_image = [target_image]
    #print('before call')
    #ret = model.is_same_id(source_image, target_image)
    ret = model.sim(source_image, target_image)
  except Exception as ex:
    print(ex)
    return '-1'

  #return str(int(ret))
  print('sim', ret)
  # return "%1.3f"%ret

def get_all_feature(ids_path=None):

  for root, dir_names, file_names in os.walk(ids_path):
    for idx, file_name in enumerate(file_names):
      if os.path.splitext(file_name)[1] not in im_extensions:
        continue
      if '_v_' in file_name:
        continue
      img_path = os.path.join(root, file_name)
      img = cv2.imread(img_path)
      target_images = model.get_all_faces(img, max_len=1333)
      for target_img in target_images:    #### can be use N batch
        _feature = model.get_feature(target_img, True)
        target_feature.append(_feature)
        ids.append(file_name)



def det_match(vedio_path=None):
  pos, all_num = 0, 0
  f = open('match_res', 'w+')
  for root, dir_names, file_names in os.walk(vedio_path):
    # all_num = len(file_names)
    for idx, file_name in enumerate(file_names):
      if os.path.splitext(file_name)[1] not in im_extensions:
        continue
      if '_i_' in file_name:
        continue
      img_path = os.path.join(root, file_name)
      img = cv2.imread(img_path)

      source_images = model.get_all_faces(img, max_len=1333)
      for idx,source_image in enumerate(source_images):
        #ret = model.is_same_id(source_image, target_image)
        source_feature = model.get_feature(source_image, True)
        cos_dist, match_idx, cos_dists = model.sim(source_feature, target_feature)
        if cos_dist < 0.38:  # cos 0.38
          continue
        if file_name.split('_')[2] == ids[match_idx].split('_')[2]:
          pos += 1
        all_num += 1
        f.write( '{}\t{}\n'.format(file_name, ids[match_idx]))
        print(file_name, ids[match_idx])
  print(pos/all_num)



if __name__ == '__main__':
    get_all_feature('/media/chen/6f586f18-792a-40fd-ada6-59702fb5dabc/data/compe/demo/images/')
    det_match('/media/chen/6f586f18-792a-40fd-ada6-59702fb5dabc/data/compe/demo/vedios/')
