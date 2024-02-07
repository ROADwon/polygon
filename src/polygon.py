import cv2 as cv
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import os
import timeit
import sklearn
import sys

from ultralytics import YOLO
from collections import Counter,deque
from shapely.geometry import Point
from shapely.geometry import Polygon
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score,classification_report


def draw_polygon(event,x,y,flags,params):
    global points
    if event == cv.EVENT_LBUTTONDOWN:
        points.append((x,y))
        print(points)
        cv.circle(frame,(x,y),3,(0,255,0),-1) #클릭 좌표 표시 (frame, (x,y), size , RGB)
        if len(points) > 1:
            cv.line(frame,points[-2], points[-1], (0,255,0),3) # 이전 점과 현재점 연결


def point_in_polygon (point,polygon):
    point = Point(point)
    polygon = Polygon(polygon)
    if polygon.contains(point):
        return "invade"
    else :
        return "not_invade"
    
def blur_img(image,kernel_size):
    blured_img = cv.blur(image, (kernel_size,kernel_size))
    return blured_img

file_save_path = f"{os.getcwd()}/confusion_matrix"
video_save_path = f"{os.getcwd()}/video_result"
csv_save_path = f"{os.getcwd()}/csv_result"
os.makedirs(video_save_path, exist_ok=True)
os.makedirs(csv_save_path, exist_ok=True)
os.makedirs(file_save_path, exist_ok=True)
frame_quality = 100
psnr_values = []

#model_path in here
#if you have a customized model or pre_trained model you can fix that
model_path = "C:/Users/line/Desktop/polygon/src"

# model start
model =   YOLO(f"{model_path}/yolov5mu.pt")# if you have a model better than yolov5m.pt you can use this sentence YOLO(f"{model_path}/weights/best.pt)

#video_path in here
video_path = "ddalbae.mp4"
frame_quality = 1
Cnt = []
ground_truth_label = []
df = pd.DataFrame(columns=['CNT','0','invade','not_invade'])
ground_truth_label = 0

for fq in range(frame_quality,30,1):
    df_con = pd.DataFrame(columns=['invade','not_invade'])
    count = 0
    
    predicted_labels = []
    cap = cv.VideoCapture(video_path)
    
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv.CAP_PROP_FPS)
    
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    output_path = f'{video_save_path}/result_{fq}.mp4'
    out = cv.VideoWriter(output_path, fourcc, fps,(width,height))
    
    points = []
    
    cv.namedWindow(f'{fq}')
    cv.setMouseCallback(f'{fq}',draw_polygon)
    #cv.setMouseCallback(f'{fq}',draw_new_polygon)
    # if you fixed the polygon h,w you can use this sentence points = [(val1,val2),(val3,val4),...,(valn,valn+1)] 
    invade_check = deque(maxlen=3)
    
    while True :
        count += 1
        invade_frame = 0
        ret,frame = cap.read()
        
        if not ret :
            break
        
        frame_c  = frame.copy()
        frame_degrade = blur_img(frame,fq)
        psnr = cv.PSNR(frame_c, frame_degrade)
        psnr_values.append(psnr)
        
        mask = np.zeros_like(frame_degrade[:,:,0])  
        if len(points) > 1:
            points_arr = np.array(points)
            cv.polylines(frame_degrade,[points_arr.astype(np.int32)], isClosed=True, color=(0,255,0), thickness=2) #draw polygon's vector, isClosed= T/F color = RGB thickness = int
            
        results = model.predict(frame_degrade,verbose=False,conf=0.5)[0]
        boxes = results.boxes.cpu().numpy()
        
        if len(points) > 3:
            class_check = []
            for box in boxes:
                
                r = box.xyxy[0].astype(int)
                ct = box.xywh[0].astype(int)
                text_position = (r[0], r[1] - 10)            
                class_name = point_in_polygon(ct[:2], points)
                cv.rectangle(frame_degrade, r[:2],r[:2],(255,255,255),2)
                confi = np.round(box.conf[0],4)
                cv.putText(frame_degrade, f"{class_name}/ {confi:.3f}", text_position, cv.FONT_HERSHEY_SIMPLEX, .9, (255, 255,255),2, cv.LINE_AA)
                class_check.append(class_name)
                predicted_labels.append(class_name)
                
                
        else :
            class_check = []
            for box in boxes :
           
                r = box.xyxy[0].astype(int)
                ct = box.xywh[0].astype(int)
                text_position = (r[0], r[1] - 10)            
                class_name = "not_invade"
                cv.rectangle(frame_degrade, r[:2],r[:2],(255,255,255),2)
                confi = np.round(box.conf[0],4)
                cv.putText(frame_degrade, f"{class_name}/ {confi:.3f}", text_position, cv.FONT_HERSHEY_SIMPLEX, .9, (255, 255,255),2, cv.LINE_AA)
                class_check.append(class_name)
                predicted_labels.append(class_name)
                
        if len(results.boxes) == 0 : 
            predicted_labels.append('0')
        
        class_counts = Counter(class_check)
        if 'invade' in class_check :
            invade_frame  = 'invade'
        invade_check.append(invade_frame)
        
        if fq == 1:
            ground_truth_label = predicted_labels
            
            
        cv.imshow(f'{fq}',frame_degrade)
        key = cv.waitKey(1) & 0xFF
        
        
        if key == ord('r') :
            points = []
            break
        
        if key == ord('q') :
            cap.release()
            out.release()
            cv.destroyAllWindows()          
            sys.exit()
        
        df_counter = pd.DataFrame.from_dict(Counter(class_check), orient= 'index', columns=['Count']).T
        df_counter = df_counter.reset_index(drop= True)
        
        df_con = pd.concat([df_con,df_counter], axis= 0)
        
        out.write(frame_degrade)
        
    cd = Counter(predicted_labels)
    df_count = pd.DataFrame({'count' : [count],
                                 'PSNR' : [sum(psnr_values)/len(psnr_values)]})
    df3 = pd.concat([df_count], axis=1)
    df = pd.concat([df,df3], axis= 0)
        
    cap.release()
    out.release()
    cv.destroyAllWindows()
    counter_csv = f'{csv_save_path}/fram_q_{fq}.csv'
    df_con = df_con.fillna(0)
    df_con.to_csv(counter_csv, index=False)
        

while os.path.exists(f'{video_save_path}/result_{vid}.mp4') :
    vid += 1
output_csv = f'{csv_save_path}/result_v{vid}.csv'
df.to_csv(output_csv, index= False)    

                
    