

from typing import List
import cv2
import torch
import numpy as np
import scipy.spatial.distance as distance
import traceback

#from MyGenerator.Bbox import Bbox


def get_center_counters_fact(input_item) -> torch.FloatTensor:
    #print(f'{type(input_item)=}')
    #print(f'{len(input_item)=}')
    tor:torch.Tensor = input_item[8]
    count = 0
    if tor.dim() == 1:
        count = sum(tor).item()
    else:
        count = sum(tor[0]).item()
    #print(f'{input_item[6].dim()=}')
    
    contours = []
    if input_item[6].dim() == 3:
        contours = input_item[6].cpu().numpy()
    else:
        contours = input_item[6][0].cpu().numpy()
        
    #print(f'{contours=}')
    
    result = []
    for i in range(count):
        contour = contours[i]
        #print(f'{contour=}')
        M = cv2.moments(contour)
        #print(f'{M=}')
        
        if M['m00'] == 0: break
        
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        result.append([cx,cy])
    return torch.FloatTensor(result)

def get_center_counters_predict(output_dict,iter = -1) -> torch.FloatTensor:
    
    py = output_dict['py_preds'][iter]
    contours = py.data.cpu().numpy()
    result = []
    for contour in contours:
        M = cv2.moments(contour)
        
        if M['m00'] == 0: break
        
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        result.append([cx,cy])
    return torch.FloatTensor(result)

mse = torch.nn.MSELoss()

def get_center_loss(tensor1,tensor2):
    '''
        except tensor1,2 shape (n1,n2)
    '''
    min_size = min(tensor1.shape[0], tensor2.shape[0])
    
    if min_size == 0: return torch.tensor([1000])
    
    mm1 = tensor1[:min_size,:]
    mm2 = tensor2[:min_size,:]
    
    res = mse(mm1,mm2)
    
    if res is None:
        res = torch.tensor([1000])
    return res 


def get_count_box_loss(tensor_predict, tensor_fact):
    res = mse(tensor_predict, tensor_fact)
    if res is None:
        res = torch.tensor([1000])
    return res 
    
def get_array_loss(tensor_predict, tensor_fact):
    '''
        except argument tensor1,2 shape (n1,n2)
        one tensor = bbox [x1, y1, x2, y2]
        s_i = (x2 - x1) * (y2 - y1)
    '''
    min_size = min(tensor_predict.shape[0], tensor_fact.shape[0])
    
    if min_size == 0: return torch.tensor([1000])
    
    mm_predict = tensor_predict[:min_size,:]
    mm_fact = tensor_fact[:min_size,:]
    
    r_predict  = ((mm_predict[...,2] - mm_predict[...,0]) * (mm_predict[...,3] - mm_predict[...,1]))
    r_fact = ((mm_fact[...,2] - mm_fact[...,0]) * (mm_fact[...,3] - mm_fact[...,1]))
    
    res = mse(r_predict, r_fact)
    if res is None:
        res = torch.tensor([1000])
    return res



def image_box_show(img, boxs, color = (255,0,0)):
    for box in boxs:
        p1,p2 = box.get_p1_p2()
        img = cv2.rectangle(np.asarray(img),p1,p2,color,1)
    return img

def image_box_show_1(img, boxs, color = (255,0,0)):
    for box in boxs:
        p1,p2 = (int(box[0]),int(box[1])),(int(box[2]),int(box[3]))
        img = cv2.rectangle(np.asarray(img),p1,p2,color,1)
    return img


cos_em_loss = torch.nn.CosineEmbeddingLoss()

def get_text_coss_loss(t1:torch.Tensor,t2:torch.Tensor):
    
    if t1.dim() == 1: t1.unsqueeze_(0)
    if t2.dim() == 1: t2.unsqueeze_(0)

    assert t1.dim() == t2.dim(), f'dim t1 and t2 can be equeal, now - {t1.dim()=}, {t2.dim()=}'
    assert t1.shape[0] == t2.shape[0], f'batch size t1 and t2 can be equeal, now - {t1.shape[0]=}, {t2.shape[0]=}'

    if t1.shape[1] != t2.shape[1]:
        min_size = min(t1.shape[1], t2.shape[1])    
        if min_size == 0: 
            return cos_em_loss.forward(torch.tensor([[1]]),torch.tensor([[0]]),torch.tensor([1]))        
        mm1 = t1[:,:min_size]
        mm2 = t2[:,:min_size]
        return cos_em_loss.forward(mm1,mm2,torch.tensor([1]))

    return cos_em_loss.forward(t1,t2,torch.tensor([1]))




# adaptation from:
# https://github.com/vigneshgig/sorting_algorthim_for_bounding_box_from_left_to_right_and_top_to_bottom/blob/master/sorting_algorthim.py
def sorting_bounding_box(points):
    '''
        format points:
        [
            ['text',[x_min,y_min,x_max,y_max] ],
            ['text',[x_min,y_min,x_max,y_max] ],
        ]

        return sorted:
        [
            [ #secuence
                ['text',[x_min,y_min,x_max,y_max] ],
                ['text',[x_min,y_min,x_max,y_max] ],
            ]
        ]
    '''
    points_start = points

    points = [[i,p[1] ]  for i,p in enumerate(points)]

    #print('-'*10)
    points = list(map(lambda x:[x[0],[x[1][0], x[1][1]], [x[1][2], x[1][3]]],points))
    points_sum = list(map(lambda x: [x[0],x[1],sum(x[1]),x[2][1]],points))
    #print(*points_sum,sep='\n')

    x_y_cordinate = list(map(lambda x: x[1],points_sum))
    final_sorted_list = []
    while True:
        try:
            if len(points_sum) == 0:
                break
            new_sorted_text = []
            #x[1][3],
            index_A, initial_value_A  = [i for i in sorted(enumerate(points_sum), key=lambda x: [x[1][2]])][0]

            x_y_min = initial_value_A[1]
            y_min = x_y_min[1]
            y_max = initial_value_A[3]
            threshold_value = abs(y_min - y_max)
            threshold_value = (threshold_value / 2) + 5
            del points_sum[index_A]
            del x_y_cordinate[index_A]
            XA = [x_y_min]
            x_y_min_other = [[ [x, y] , abs(y - y_min)] for x,y in x_y_cordinate]            
            x_y_min_other = [[count,i] for count, i in enumerate(x_y_min_other)]
            x_y_min_other = [i for i in x_y_min_other if i[1][1] <= threshold_value] # threshold by abs(y - y_min)
            sorted_K = [[x[0], x[1][0]] for x in sorted(x_y_min_other, key=lambda x:x[1][1])]
            XB = []
            points_index = []
            for tmp_K in sorted_K:
                points_index.append(tmp_K[0])
                XB.append(tmp_K[1])
            d_index = []
            if len(XB) > 0:                
                dist = distance.cdist(XA,XB)[0]
                d_index = [i for i in sorted(zip(dist, points_index), key=lambda x:x[0])]
            new_sorted_text.append(initial_value_A[0])
            #print(new_sorted_text)

            index = []
            for j in d_index:
                new_sorted_text.append(points_sum[j[1]][0])
                index.append(j[1])
            for n in sorted(index, reverse=True):
                del points_sum[n]
                del x_y_cordinate[n]
            final_sorted_list.append(new_sorted_text)
        except Exception as e:
            traceback.print_exc()
            break
    #print('-'*10)
    #print(points_start)
    #post final    
    points2 = []
    for bbox in final_sorted_list:
        points1 = []
        for idx in bbox:
            points1.append(points_start[idx])
        points2.append(points1)
    #print(points1)

    return points2#,final_sorted_list



def sort_by_x(lines_list):
    lines_list_new = []
    for line in lines_list:
        line = sorted(line, key= lambda x:x[1][0]) 
        lines_list_new.append(line)
    return lines_list_new
        
        

def get_avg_hight(points):
    a = np.quantile([box[3] - box[1] for text,box in points],0.8)
    return a

def sort_by_y(points):    
    avg_hight = get_avg_hight(points)
    points = sorted(points,key = lambda x:x[1][1]) #sort by y_min
    res_all = []
    res = [points[0]]
    for point_index in range(1,len(points)):
        point = points[point_index]
        text,box = point
        x_min,y_min,x_max,y_max = box
        hight = y_max - y_min
        max_from_min_y = max([x[1][1] for x in res])
        max_from_max_y = max([x[1][3] for x in res])
        diff_size_top = abs(y_min - max_from_min_y)
        diff_size_s = abs(y_min - max_from_max_y)
        #p = diff_size_s/avg_hight
        p1 = diff_size_s/hight
        
        if diff_size_top < avg_hight and p1 > 0.4:
            res.append(point)
        else:
           res_all.append(res)
           res = [point]    
    res_all.append(res)    
    return res_all

def sorting_bounding_box_v2(points):
    lines_list = sort_by_y(points)
    lines_list = sort_by_x(lines_list)    
    return lines_list



def get_min_left_box_pix_size_step(points):
    min_left = np.inf
    it = 0
    one_steps = []
    for point_list in points:
        for text, box in point_list:
            it += 1
            if box[0] < min_left:
                min_left = box[0]
            h = box[2] - box[0]
            
            one_steps.append(h / len(text))
            # one_step += h / len(text)
            # one_step /= it
    return min_left,np.mean(one_steps)

def get_text_from_points(points):
    res = []
    
    min_left_px, one_step_size_px = get_min_left_box_pix_size_step(points)

    for point_list in points:
        is_first = True
        text_line = ''        
        for text, box in point_list:
            x_min,x_max = box[0],box[2]

            if not is_first:
                text_line += ' '            
            if is_first:
                if one_step_size_px == 0:
                    h = x_max - x_min
                    one_step_size_px = h / len(text)
                position = x_min - min_left_px
                if position >= one_step_size_px:
                    steps = int((x_min - min_left_px)/one_step_size_px)
                    text = ' ' * steps + text
            is_first = False
            text_line += text
        res.append(text_line)
    return res

def convert_to_index_box(bboxs):
    '''
        convert from 
        [
            [x_min,y_min,x_max,y_max],
            [x_min,y_min,x_max,y_max]
        ]
        to
        [
            [0,[x_min,y_min,x_max,y_max]],
            [1,[x_min,y_min,x_max,y_max]]
        ]
    '''
    res = []
    for i,box in enumerate(bboxs):        
        res.append([i,box])
    
    return res



if __name__ == '__main__':
    print(f'{get_text_coss_loss(torch.IntTensor([1,2,3]),torch.IntTensor([1,2,3]))}')
    print(f'{get_text_coss_loss(torch.IntTensor([[1,2,3]]),torch.IntTensor([[1,2,3]]))}')
    print(f'{get_text_coss_loss(torch.IntTensor([[]]),torch.IntTensor([[1,2,3]]))}')
    print(f'{get_text_coss_loss(torch.IntTensor([[1,2,3,4]]),torch.IntTensor([[1,2,3]]))}')
    print(f'{get_text_coss_loss(torch.IntTensor([[1,2,2]]),torch.IntTensor([[1,2,3]]))}')
    
