from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import sys
import os
import argparse
import numpy as np
import mxnet as mx
import random
import sklearn
from sklearn.decomposition import PCA
from time import sleep
from easydict import EasyDict as edict
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'common'))
import face_image
# import face_preprocess
import torch, cv2
sys.path.append(os.path.join(os.path.dirname(__file__), 'mmdet'))
from mmdet.apis import inference_detector, init_detector, show_result

def ch_dev(arg_params, aux_params, ctx):
  new_args = dict()
  new_auxs = dict()
  for k, v in arg_params.items():
    new_args[k] = v.as_in_context(ctx)
  for k, v in aux_params.items():
    new_auxs[k] = v.as_in_context(ctx)
  return new_args, new_auxs

def do_flip(data):
  for idx in range(data.shape[0]):
    data[idx,:,:] = np.fliplr(data[idx,:,:])

class FaceModel:
  def __init__(self, args):
    model = edict()
    self.det = init_detector(
        args.det_config, args.det_checkpoint, device=torch.device('cuda', args.det_device))

    self.threshold = args.threshold
    self.det_minsize = 50
    self.det_threshold = [0.4,0.6,0.6]
    self.det_factor = 0.9
    _vec = args.image_size.split(',')
    assert len(_vec)==2
    self.image_size = (int(_vec[0]), int(_vec[1]))
    _vec = args.model.split(',')
    assert len(_vec)==2
    prefix = _vec[0]
    epoch = int(_vec[1])
    print('loading',prefix, epoch)
    self.model = edict()
    self.model.ctx = mx.gpu(args.gpu)
    self.model.sym, self.model.arg_params, self.model.aux_params = mx.model.load_checkpoint(prefix, epoch)
    self.model.arg_params, self.model.aux_params = ch_dev(self.model.arg_params, self.model.aux_params, self.model.ctx)
    all_layers = self.model.sym.get_internals()
    self.model.sym = all_layers['fc1_output']

  def resize_image_fix(self, im_size, max_len=1600):
    h, w = im_size
    f = float(max_len) / max(w, h)
    o_height = int(f * h)
    o_width = int(f * w / 1)
    d_width = o_width - (o_width % 32)
    d_height = o_height - (o_height % 32)
    return d_width, d_height

  def image_resize(self, img):
    h, w, c = img.shape
    img_dst = np.ones((224, 224, 3), dtype=np.uint8) * 128
    if h >= w:
      dst_h = 224;
      dst_w = int(w / h * 224.0)
    else:
      dst_w = 224;
      dst_h = int(h / w * 224.0)
    img = cv2.resize(img, (dst_w, dst_h))
    img_dst[int((224 - dst_h) / 2):int((224 - dst_h) / 2) + dst_h, \
            int((224 - dst_w) / 2):int((224 - dst_w) / 2) + dst_w, :] = img
    # img_dst = img_dst.transpose((2, 0, 1))
    return img_dst

  def get_all_faces(self, img, max_len=1600):
    str_image_size = "%d,%d"%(self.image_size[0], self.image_size[1])
    h, w = img.shape[:2]
    d_w ,d_h = self.resize_image_fix((h,w), max_len)
    img_det = cv2.resize(img, (d_w, d_h))
    det_ratio =  (np.array([h, w])/np.array(img_det.shape[:2]))[::-1]
    result = inference_detector(self.det, img_det)
    if isinstance(result, tuple):
      bbox_result, segm_result = result
    else:
      bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)
    labels = [
      np.full(bbox.shape[0], i, dtype=np.int32)
      for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    score_thr = 0.3
    if score_thr > 0:
      assert bboxes.shape[1] == 5
      scores = bboxes[:, -1]
      inds = scores > score_thr
      bboxes = bboxes[inds, :]
      labels = labels[inds]
    bboxes[:,::2] = bboxes[:,::2] * det_ratio[0]
    bboxes[:, 1::2] = bboxes[:, 1::2] * det_ratio[1]

    ret = []
    for bbox, label in zip(bboxes, labels):
      bbox_int = bbox.astype(np.int32)

      left_top = (bbox_int[0], bbox_int[1])
      right_bottom = (bbox_int[2], bbox_int[3])
      aligned = img[bbox_int[1]:min(bbox_int[3],h), bbox_int[0]:min(bbox_int[2],w), :]
      # landmark = points[:, i].reshape((2,5)).T
      # aligned = face_preprocess.preprocess(img, bbox=bbox, landmark = landmark, image_size=str_image_size)
      aligned = aligned[:,:,::-1] #cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
      aligned = self.image_resize(aligned)
      aligned = np.transpose(aligned, (2,0,1))
      ret.append(aligned)
    return ret

  def get_feature_impl(self, face_img, norm):
    embedding = None
    for flipid in [0,1]:
      _img = np.copy(face_img)
      if flipid==1:
        do_flip(_img)
      #nimg = np.zeros(_img.shape, dtype=np.float32)
      #nimg[:,ppatch[1]:ppatch[3],ppatch[0]:ppatch[2]] = _img[:, ppatch[1]:ppatch[3], ppatch[0]:ppatch[2]]
      #_img = nimg
      input_blob = np.expand_dims(_img, axis=0)
      self.model.arg_params["data"] = mx.nd.array(input_blob, self.model.ctx)
      self.model.arg_params["softmax_label"] = mx.nd.empty((1,), self.model.ctx)
      exe = self.model.sym.bind(self.model.ctx, self.model.arg_params ,args_grad=None, grad_req="null", aux_states=self.model.aux_params)
      exe.forward(is_train=False)
      _embedding = exe.outputs[0].asnumpy()
      #print(_embedding.shape)
      if embedding is None:
        embedding = _embedding
      else:
        embedding += _embedding
    if norm:
      embedding = sklearn.preprocessing.normalize(embedding)
    return embedding

  def get_feature(self, face_img, norm=True):
    #aligned_face = self.get_aligned_face(img, force)
    #if aligned_face is None:
    #  return None
    return self.get_feature_impl(face_img, norm)
  #
  # def is_same_id(self, source_img, target_img_list):
  #   source_face = self.get_aligned_face(source_img, True)
  #   print('source face', source_face.shape)
  #   target_face_list = []
  #   pp = 0
  #   for img in target_img_list:
  #     target_force = False
  #     if pp==len(target_img_list)-1 and len(target_face_list)==0:
  #       target_force = True
  #     target_face = self.get_aligned_face(img, target_force)
  #     if target_face is not None:
  #       target_face_list.append(target_face)
  #     pp+=1
  #   print('target face', len(target_face_list))
  #   source_feature = self.get_feature(source_face, True)
  #   target_feature = None
  #   for target_face in target_face_list:
  #     _feature = self.get_feature(target_face, False)
  #     if target_feature is None:
  #       target_feature = _feature
  #     else:
  #       target_feature += _feature
  #   target_feature = sklearn.preprocessing.normalize(target_feature)
  #   #sim = np.dot(source_feature, target_feature.T)
  #   diff = np.subtract(source_feature, target_feature)
  #   dist = np.sum(np.square(diff),1)
  #   print('dist', dist)
  #   #print(sim, dist)
  #   if dist<=self.threshold:
  #     return True
  #   else:
  #     return False

  def sim(self, source_img, target_img_list):
    print('sim start')
    source_face = source_img #self.get_aligned_face(source_img, True)
    print('source face', source_face.shape)
    target_face_list = []
    pp = 0
    for img in target_img_list:
      target_force = False
      if pp==len(target_img_list)-1 and len(target_face_list)==0:
        target_force = True
      target_face = img #self.get_aligned_face(img, target_force)
      if target_face is not None:
        target_face_list.append(target_face)
      pp+=1
    print('target face', len(target_face_list)) 
    source_feature = self.get_feature(source_face, True)
    target_feature = None
    sim_list = []
    for target_face in target_face_list:
      _feature = self.get_feature(target_face, True)
      _sim = np.dot(source_feature, _feature.T)
      sim_list.append(_sim)
    return np.max(sim_list), np.argmax(sim_list), np.array(sim_list)

  def sim(self, source_feature, target_featue_list):
    # source_face = source_img #self.get_aligned_face(source_img, True)
    # print('source face', source_face.shape)
    # target_face_list = []
    # pp = 0
    # for img in target_img_list:
    #   target_force = False
    #   if pp==len(target_img_list)-1 and len(target_face_list)==0:
    #     target_force = True
    #   target_face = img #self.get_aligned_face(img, target_force)
    #   if target_face is not None:
    #     target_face_list.append(target_face)
    #   pp+=1
    # print('target face', len(target_face_list))
    # source_feature = self.get_feature(source_face, True)
    target_feature = None
    sim_list = []
    for _feature in target_featue_list:
      # _feature = self.get_feature(target_face, True)
      _sim = np.dot(source_feature, _feature.T)
      sim_list.append(_sim)
    return np.max(sim_list), np.argmax(sim_list), np.array(sim_list)
