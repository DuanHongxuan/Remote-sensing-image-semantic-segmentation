import cv2 as cv
import numpy as np
from collections import Counter
import xlwt as excel
from decimal import Decimal
import os
import skimage.io as io
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def cal_confu_matrix(label, predict, class_num):
    confu_list = []
    for i in range(class_num):
        c = Counter(predict[np.where(label == i)])
        single_row = []
        for j in range(class_num):
            single_row.append(c[j])
        confu_list.append(single_row)
    return np.array(confu_list).astype(np.int32)

def metric_evaluate(confu_mat_total, save_path, save_name):
    
    class_num = confu_mat_total.shape[0]
    file = excel.Workbook(encoding='utf-8')
    table_name = save_name
    pic_name = table_name + ' metrics:'
    table = file.add_sheet(table_name)
    table_raw = 0  # first begin
    table.write(table_raw, 0, pic_name)
    table_raw += 2
    '''output confu_mat to Excel'''
    confu_mat = confu_mat_total.astype(np.float32) + 0.0001
    col_sum = np.sum(confu_mat, axis=1)   # sum col
    raw_sum = np.sum(confu_mat, axis=0)   # sum raw
    
    oa = 0
    for i in range(class_num):
        oa = oa + confu_mat[i, i]
    oa = oa / confu_mat.sum()
    '''Kappa'''
    pe_fz = 0
    for i in range(class_num):
        pe_fz += col_sum[i] * raw_sum[i]
    pe = pe_fz / (np.sum(confu_mat) * np.sum(confu_mat))
    kappa = (oa - pe) / (1 - pe)
    # excel
    TP = []
    table.write(table_raw, 0, 'confusion_matrix:')
    table_raw = table_raw + 1
    #name_str = ['Clutter/background', 'Impervious surfaces', 'Building', 'Low vegetation', 'Tree', 'Car']
    name_str = ['Impervious surfaces','Building','Low vegetation','Tree','Car']
    for i in range(class_num):
        table.write(table_raw, 1+i, name_str[i])
    for i in range(class_num):
        table_raw = table_raw + 1
        table.write(table_raw, 0, name_str[i])
        TP.append(confu_mat[i, i])
        for j in range(class_num):
            table.write(table_raw, j + 1, int(confu_mat_total[i, j]))
    # f1-score
    TP = np.array(TP)
    FN = raw_sum - TP
    FP = col_sum - TP
    # precision recall, f1-score f1-m mIOU
    table_raw = table_raw + 2
    table.write(table_raw, 0, 'precision:')
    # precision
    for i in range(class_num):
        table.write(table_raw, i + 1, Decimal(float(TP[i]/raw_sum[i])).quantize(Decimal("0.0000")))
    table_raw += 1
    table.write(table_raw, 0, 'Recall:')
    # recall
    for i in range(class_num):
        table.write(table_raw, i + 1, Decimal(float(TP[i]/col_sum[i])).quantize(Decimal("0.0000")))
    f1_m = []
    iou_m = []
    table_raw += 1
    table.write(table_raw, 0, 'f1-score:')
    for i in range(class_num):
        # f1-score
        f1 = TP[i] * 2 / (TP[i] * 2 + FP[i] + FN[i])
        f1_m.append(f1)
        iou = TP[i] / (TP[i] + FP[i] + FN[i])
        iou_m.append(iou)
        table.write(table_raw, i + 1, Decimal(f1).quantize(Decimal("0.0000")))
    table_raw += 2
    table.write(table_raw, 0, 'OA:')
    table.write(table_raw, 1, Decimal(float(oa)).quantize(Decimal("0.0000")))
    table_raw += 1
    table.write(table_raw, 0, 'Kappa:')
    table.write(table_raw, 1, Decimal(float(kappa)).quantize(Decimal("0.0000")))
    f1_m = np.array(f1_m)
    table_raw += 1
    table.write(table_raw, 0, 'f1-m:')
    table.write(table_raw, 1, Decimal(float(np.mean(f1_m))).quantize(Decimal("0.0000")))
    iou_m = np.array(iou_m)
    table_raw += 1
    table.write(table_raw, 0, 'mIOU:')
    table.write(table_raw, 1, Decimal(float(np.mean(iou_m))).quantize(Decimal("0.0000")))
    file.save(save_path + table_name + '.xls')


if __name__=="__main__":    
    #path1 = "/storage/student17/rs/data/source/Vaihingen/evalution_new" #Dir of Ground Truth
    #path2 = "/storage/student17/rs/predict/12-1"  #Dir of predict map
    path1 = "/storage/student17/rs/data/source/Potsdam/test/new_label"
    path2 = "/storage/student17/rs/predict/17-1"
    save_path = "/storage/student17/rs/predict/17-2/"
    
    sample1 = os.listdir(path1)

    for name in sample1:
        print("name:",name)
        label = cv.imread(os.path.join(path1, name))
        label = np.array(label)
        #label = label / 255.0
        label = label.flatten()

        predict = cv.imread(os.path.join(path2, name))
        predict = np.array(predict)
        #predict = predict / 255.0
        predict = predict.flatten()

        #confusion = confusion_matrix(label, predict)
        confusion = cal_confu_matrix(label, predict, 5)
        print("confusion:",confusion)
        save_name = name
        metric_evaluate(confusion, save_path, save_name)
