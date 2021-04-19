import os
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing
from functools import partial
import cv2
import time
import imageio
from . import metric

def get_files(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

def get_folders(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isdir(os.path.join(folder, f))]

def parse_key(key):
    """ key -> num, size
    """
    num = key[key.find('num_') + len('num_'):key.find('_size_')]
    size = key[key.find('size_') + len('size_'):key.find('_ibl')]
    return int(num), float(size)

def compute_metric(pred_np, mts_np):
    pred_mts = [metric.rmse(pred_np, mts_np), metric.rmse_s(pred_np, mts_np)[1], metric.ZNCC(pred_np, mts_np)]
    return np.array(pred_mts)

def get_predict(files):
    predict_f = ''
    for f in files:
        if f.find('_predict.png') != -1:
            predict_f = f

    return predict_f

def get_mits(files):
    mts_f = ''
    for f in files:
        if f.find('_shadow.exr') != -1:
            mts_f = f

    return mts_f

# parallel version
def worker(input_param):
    ibl_key, ibl_dict, model_list, exp_model_list = input_param
    sample_num = len(ibl_dict[ibl_key]) * len(model_list)
    cur_metric = np.zeros((1,3))
    for ibl_folder in ibl_dict[ibl_key]: 
        for i, m in enumerate(model_list):
            cur_folder = join(m, join('pattern', ibl_folder))
            exp_cur_folder = join(exp_model_list[i], join('pattern', ibl_folder))
            
            # pred_path, mitsuba_path = join(cur_folder, 'predict.png'), join(cur_folder, 'mitsuba_shadow.exr')
            pred_path, mitsuba_path = get_predict(get_files(exp_cur_folder)), get_mits(get_files(cur_folder))
            pred_np = plt.imread(pred_path)[:,:,0]
            
            if mitsuba_path == '':
                # print('cannot find mitsuba rendering result: ', cur_folder)
                mts_np = np.zeros((256,256))
            else:
                mts_np = 1.0 - imageio.imread(mitsuba_path, format='exr')[:,:,0]
            
            # pred_np -> mts_np(mse, mses, zncc)
            # average over all samples
            pred_mts = compute_metric(pred_np, mts_np)
            cur_metric[0] += pred_mts / sample_num
    return (ibl_key, cur_metric)

def parallel_metric(model_list, exp_model_list, metric_dict, processor_num = 72):
    s = time.time()
    
    # (pred) x (mse, mse_s, zncc)
    metric_result = {}
    for k in metric_dict.keys():
        metric_result[k] = np.zeros((1,3))
    
    task_num = len(metric_dict.keys())
    input_list = zip(metric_dict.keys(), [metric_dict] * task_num, [model_list] * task_num, [exp_model_list] * task_num)
    
    with multiprocessing.Pool(processor_num) as pool:
        # working_fn = partial(batch_working_process, src_folder, out_folder)
        for i, cur_result in enumerate(pool.imap_unordered(worker, input_list), 1):
            ibl_key, cur_metric = cur_result
            metric_result[ibl_key] = cur_metric
            
            print("Finished: {} \r".format(float(i)/task_num), flush=True, end='')
    
    print('metric computation time: {} s'.format(time.time() - s))
    return metric_result

def compute_exp_results(eval_folder, exp_out_folder):
    model_list = get_folders(eval_folder)
    exp_model_list = get_folders(exp_out_folder)
    
    model_list.sort(); exp_model_list.sort()

    # prepare an ibl list
    model = model_list[0]
    ibl_folder = join(model, 'pattern')
    ibl_folders = get_folders(ibl_folder)
    
    ibls = set()
    for ibl_f in ibl_folders:
        ibl_name = os.path.basename(ibl_f)
        ibls.add(ibl_name)

    # compute metric experiment results
    ibl_num_dict, ibl_size_dict = dict(), dict()
    for ibl in tqdm(ibls):
        num, size = parse_key(ibl)
    
        if num not in ibl_num_dict.keys():
            ibl_num_dict[num] = []
            
        if size not in ibl_size_dict.keys():
            ibl_size_dict[size] = []
    
        ibl_num_dict[num].append(ibl)
        ibl_size_dict[size].append(ibl)

    num_metric_result = parallel_metric(model_list, exp_model_list, ibl_num_dict)
    size_metric_result = parallel_metric(model_list, exp_model_list, ibl_size_dict)

    return num_metric_result,size_metric_result