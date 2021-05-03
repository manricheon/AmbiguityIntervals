import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from scipy import io
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def load_image(path):
  return np.array(Image.open(path).convert('RGB'))

def get_data(arr_from_mat):
  no_ref = np.shape(arr_from_mat)[1]
  data = []
  for idx in range(no_ref):
    _data = arr_from_mat[0][idx]
    data.append(_data)
  return data

def get_ref_names(arr_from_mat):
  no_ref = np.shape(arr_from_mat)[1]
  ref_names = []
  for idx in range(no_ref):
    _ref_name = arr_from_mat[0][idx]
    ref_names.extend(_ref_name)
  return ref_names

def get_dist_types(arr_from_mat):
  no_types = np.shape(arr_from_mat)[1]
  dist_types = []
  for idx in range(no_types):
    _dist_type = arr_from_mat[0][idx]
    dist_types.extend(_dist_type)
  return dist_types

# you can change a quality metric
def measure_quality(ref_img, dist_path, dist_list):
  dist_imgs = []
  q_scores = []
  for i in range(len(dist_list)):
    _dist_img = load_image(dist_path + dist_list[i])
    dist_imgs.append(_dist_img)

    _q_score = ssim(ref_img, _dist_img, multichannel=True)
    q_scores.append(_q_score)

  return dist_imgs, q_scores


# just example (based on difference information)
# recommend to use vdp model
def JND_model(img1, img2):
  img = img1-img2
  img = np.mean(img, axis=-1)
  img = np.abs(img) / np.max(img + 1e-6)
  return img


def measure_ambiguity_interval(ref_img, dist_imgs, q_scores, vdp_info, threshold, N, is_plot=False):

  qs = []   # quality score
  qs_u = [] # upper quality score
  qs_l = [] # lower quality score
  qs_u_intv = []  # upper quality interval 
  qs_l_intv = []  # lower quality interval
  qs_intv = []

  for i in range(N):

    # lower interval
    l_intv = 0.
    l_qs = q_scores[i]
    for j in range(i,N):

      # use JND model and calculate percentage
      # jnd_map = JND_model(dist_imgs[i], dist_imgs[j])
      # jnd_map_h, jnd_map_w = np.shape(jnd_map)[:2]
      # perceivableness_map = jnd_map[jnd_map>0.5]
      # PmapCount = np.sum(perceivableness_map) / (jnd_map_h * jnd_map_w)

      # use example vdp info (percentage)
      PmapCount = vdp_info[i][j]

      if PmapCount >= threshold:
        l_qs = q_scores[j]
        l_intv = q_scores[i] - q_scores[j]
        break


    # upper interval
    u_intv = 0.
    u_qs = q_scores[i]
    for k in range(i, 0, -1):

      # # use JND model 
      # jnd_map = JND_model(dist_imgs[i], dist_imgs[k])
      # jnd_map_h, jnd_map_w = np.shape(jnd_map)[:2]
      # perceivableness_map = jnd_map[jnd_map>0.5]  
      # PmapCount = np.sum(perceivableness_map) / (jnd_map_h * jnd_map_w)

      # use example vdp info (percentage)
      PmapCount = vdp_info[i][k]

      if PmapCount >= threshold:
        u_qs = q_scores[k]
        u_intv = q_scores[k] - q_scores[i]
        break

    qs.append(q_scores[i])
    qs_u.append(u_qs)
    qs_l.append(l_qs)
    qs_u_intv.append(u_intv)
    qs_l_intv.append(l_intv)
    qs_intv.append(u_intv + l_intv)

  # mean_amb_intv = np.mean(qs_u_intv + qs_l_intv) / (np.max(qs) - np.min(qs))
  # max_amb_intv = np.max(qs_u_intv + qs_l_intv) / (np.max(qs) - np.min(qs))
  # min_amb_intv = np.min(qs_u_intv + qs_l_intv) / (np.max(qs) - np.min(qs))  

  mean_amb_intv = np.mean(qs_intv) / (np.max(qs) - np.min(qs))
  max_amb_intv = np.max(qs_intv) / (np.max(qs) - np.min(qs))
  min_amb_intv = np.min(qs_intv) / (np.max(qs) - np.min(qs))  

  print('mean, max, min of ambiguity interval :', mean_amb_intv, max_amb_intv, min_amb_intv)

  if is_plot:
    plt.plot(qs, qs)
    plt.fill_between(qs, qs_u, qs_l, alpha=.5)
    plt.xlabel('score')
    plt.ylabel('score')
    # plt.title('bikes.bmp - dist. type : gb')
    plt.show()

  # return qs_intv


def main():

  db_name = 'live'
  content_name = 'bikes'  # 'building2'
  dist_type = 'gb'  # 'wn'
  N = 100           # no. of distortion levels
  threshold = 0.5   # threshold for decision of distinguished images

  # set database path
  db_path = './dataset'

  # saved vdp info (.mat)
  mat_path = db_path + '/' + db_name + '/' + 'database_vdp_info_' + db_name + '.mat'
  mat_file = io.loadmat(mat_path)

  db_name = mat_file['db_name'][0]
  dist_types = get_dist_types(mat_file['artifact_name'])  # get distortion types
  ref_names = get_ref_names(mat_file['ref_image_name'])   # get reference image names
  v_distances = ['v1','v2']

  # for-loop here if multiple distortions are considered
  if content_name == 'gb':
    dist_type = dist_types[2]   # idx 0-3 : jpeg, jpeg2k, gb, wn (live dataset)
  elif content_name == 'wn':
    dist_type = dist_types[3]

  # for-loop here if multiple contents are considered
  ref_name = ref_names[0]
  v_distance = v_distances[0]

  # 1-1) use vdp info from saved mat file (just example)
  _vdp_info = 'vdp_' + dist_type + '_' + v_distance
  vdp_info = mat_file[_vdp_info][0][0]  # last index : ref_name # vdp info of bikes.bmp ; 100x100 matrix
  # vdp_info = mat_file[_vdp_info][0][1]  # last index : ref_name # vdp info of bikes.bmp ; 100x100 matrix
  # vdp_info = mat_file[_vdp_info][0][2]  # last index : ref_name # vdp info of bikes.bmp ; 100x100 matrix  
  # vdp_info = mat_file[_vdp_info][0][N-1]  # last index : ref_name # vdp info of bikes.bmp ; 100x100 matrix  

  # or
  # 1-2) use JND model from source (e.g., vdp 2.2)


  # directory of reference
  ref_path = db_path + '/' + db_name + '/' + 'ref' + '/' 
  ref_img_path = ref_path + ref_name + '.bmp'
  ref_img = load_image(ref_img_path)

  # directory of dist images (depending on dist types)
  dist_path = db_path + '/' + db_name + '/'

  # directory of image N (including distorted images)
  # imgN_qp001.bmp to imgN_qp100.bmp ; i is more degraded than i-1
  dist_paths = dist_path + dist_type + '/' + content_name + '/'
  dist_list = sorted(os.listdir(dist_paths), reverse=False)

  # read dist imgs and measure quality (cc: measure_quality function)
  dist_imgs, q_scores = measure_quality(ref_img, dist_paths, dist_list)

  # measure ambiguity interval based on vdp info and quality scores for each content
  measure_ambiguity_interval(ref_img, dist_imgs, q_scores, vdp_info, threshold, N, is_plot=True)


if __name__ == '__main__':
  main()