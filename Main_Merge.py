#merged @omarsayed7 and DeepLearning_by_PhDScholar's implementation

from __future__ import print_function
import argparse
import numpy  as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms

from data_loaders import Plain_Dataset, eval_data_dataloader
from deep_emotion import Deep_Emotion
from generate_data import Generate_data

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def Train(epochs,train_loader,val_loader,criterion,optmizer,device):
    '''
    Training Loop
    '''
    print("===================================Start Training===================================")
    for e in range(epochs):
        train_loss = 0
        validation_loss = 0
        train_correct = 0
        val_correct = 0
        # Train the model  #
        net.train()
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            optmizer.zero_grad()
            outputs = net(data)
            loss = criterion(outputs,labels)
            loss.backward()
            optmizer.step()
            train_loss += loss.item()
            _, preds = torch.max(outputs,1)
            train_correct += torch.sum(preds == labels.data)
                  
         #validate the model#
        net.eval()   
        for data,labels in val_loader:
            data, labels = data.to(device), labels.to(device)
            val_outputs = net(data)
            val_loss = criterion(val_outputs, labels)
            validation_loss += val_loss.item()
            _, val_preds = torch.max(val_outputs,1)
            val_correct += torch.sum(val_preds == labels.data)

        train_loss = train_loss/len(train_dataset)
        train_acc = train_correct.double() / len(train_dataset)
        validation_loss =  validation_loss / len(validation_dataset)
        val_acc = val_correct.double() / len(validation_dataset)
        print('Epoch: {} \tTraining Loss: {:.8f} \tValidation Loss {:.8f} \tTraining Acuuarcy {:.3f}% \tValidation Acuuarcy {:.3f}%'
                                                           .format(e+1, train_loss,validation_loss,train_acc * 100, val_acc*100))

    torch.save(net.state_dict(),'deep_emotion-{}-{}-{}.pt'.format(epochs,batchsize,lr))

    print("===================================Training Finished===================================")
    
generate_dataset = Generate_data("data//")
generate_dataset.split_test()
generate_dataset.save_images()
generate_dataset.save_images('finaltest')
generate_dataset.save_images('val')

epochs = 100
lr = 0.005
batchsize = 128


net = Deep_Emotion()
net.to(device)
print("Model archticture: ", net)
traincsv_file = 'data'+'/'+'train.csv'  #remove
validationcsv_file = 'data'+'/'+'val.csv'
train_img_dir = 'data'+'/'+'train/'
validation_img_dir = 'data'+'/'+ 'val/'

transformation= transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])
train_dataset= Plain_Dataset(csv_file=traincsv_file, img_dir = train_img_dir, datatype = 'train', transform = transformation)
validation_dataset= Plain_Dataset(csv_file=validationcsv_file, img_dir = validation_img_dir, datatype = 'val', transform = transformation)
train_loader= DataLoader(train_dataset,batch_size=batchsize,shuffle = True,num_workers=0)
val_loader=   DataLoader(validation_dataset,batch_size=batchsize,shuffle = True,num_workers=0)

criterion= nn.CrossEntropyLoss()
optmizer= optim.Adam(net.parameters(),lr= lr)
Train(epochs, train_loader, val_loader, criterion, optmizer, device)

torch.save(net.state_dict(), 'new_Emotion_trained_Lea.pt')     

net = Deep_Emotion()
net.load_state_dict(torch.load('new_Emotion_trained_Lea.pt'))
net.to(device)

# Test it on a saved image:

import matplotlib.pyplot as plt

get_ipython().system('pip install opencv-python')

import cv2

frame = cv2.imread("C:\\Users\\kassa\\11-test\\Deep-Emotion-master\\happy.jpg")

get_ipython().system('pip install deepface')

from deepface import DeepFace

plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = faceCascade.detectMultiScale(gray, 1.1, 4)
for x,y,w,h in faces:
    roi_gray = gray [y:y+h, x:x+w]
    roi_color = frame[y:y+h, x:x+w]
    cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
    facess = faceCascade.detectMultiScale(roi_gray)
    if len(facess) == 0:
        print("Face not detected")
    else:
        for (ex,ey,ew,eh) in facess:
            face_roi = roi_color[ey: ey+eh, ex: ex+ew]
plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
plt.imshow(face_roi)
gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
final_image = cv2.resize(gray, (48,48))
final_image = np.expand_dims(final_image, axis = 0)
final_image = np.expand_dims(final_image, axis = 0)
final_image = final_image/255.0
dataa = torch.from_numpy(final_image)
dataa = dataa.type(torch.FloatTensor)
dataa = dataa.to(device)
outputs = net(dataa)
pred = F.softmax(outputs, dim = 1)
print(torch.argmax(pred))
index_pred = torch.argmax(pred)
#if (index_pred == 5):
    print('Just checking the values')

import cv2
#pip install opencv-python
#pip install opencv-contriv-python
get_ipython().system('pip install deepface')
from deepface import DeepFace


# Live Demo
path = "haarcascade_frontalface_default.xml"
font_scale = 1.5
font = cv2.FONT_HERSHEY_PLAIN

# set rectangle background to white
rectangle_bgr = (255,255,255)
#make a black image
img = np.zeros((500,500))
#set some text
text = "Some text in a box!"
#get the width and height of the text box
(text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]
#set the text start position
text_offset_x = 10
text_offset_y = img.shape[0] - 25
#make the coords of the box with a small padding of two pixels
box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height - 2))
cv2.rectangle(img, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)
cv2.putText(img, text, (text_offset_x, text_offset_y), font, fontScale = font_scale, color = (0,0,0), thickness = 1)

cap = cv2.VideoCapture(1)
#check if the webcam is opened correctly
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcame")
    
while True:
    ret,frame = cap.read()
    #eye_cascade
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,1.1,4)
    for x,y,w,h in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
        facess = faceCascade.detectMultiScale(roi_gray)
        if len(facess) == 0:
            print("Face not detected")
        else:
            for (ex,ey,ew,eh) in facess:
                face_roi = roi_color[ey:ey+eh, ex:ex+ew]
    
    graytemp = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    final_image = cv2.resize(graytemp, (48,48))
    final_image = np.expand_dims(final_image, axis = 0)
    final_image = np.expand_dims(final_image, axis = 0)
    final_image = final_image/255.0
    dataa = torch.from_numpy(final_image)
    dataa = dataa.type(torch.FloatTensor)
    dataa = dataa.to(device)
    outputs = net(dataa)
    Pred = F.softmax(outputs, dim = 1)
    Predictions = torch.argmax(Pred)
    print(Predictions)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    font = cv2.FONT_HERSHEY_PLAIN
    
    if ((Predictions)==0):
        status = "Angry"
        x1,y1,w1,h1 = 0,0,175,75
        #black background
        cv2.rectangle(frame, (x1,x1), (x1+w1, y1+h1), (0,0,0), -1)
        #text
        cv2.putText(frame, status, (x1+int(w1/10), y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0, 255), 2)
        cv2.putText(frame, status, (100,150), font, 3 , (0,0, 255), 2, cv2.LINE_4)
        cv2.rectangle(frame,(x,y), (x+w, y+h), (0,0,255))
        
    elif ((Predictions)==1):
        status = "Disgust"
        x1,y1,w1,h1 = 0,0,175,75
        #black background
        cv2.rectangle(frame, (x1,x1), (x1+w1, y1+h1), (0,0,0), -1)
        #text
        cv2.putText(frame, status, (x1+int(w1/10), y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0, 255), 2)
        cv2.putText(frame, status, (100,150), font, 3 , (0,0, 255), 2, cv2.LINE_4)
        cv2.rectangle(frame,(x,y), (x+w, y+h), (0,0,255))
        
    elif ((Predictions)==2):
        status = "Fear"
        x1,y1,w1,h1 = 0,0,175,75
        #black background
        cv2.rectangle(frame, (x1,x1), (x1+w1, y1+h1), (0,0,0), -1)
        #text
        cv2.putText(frame, status, (x1+int(w1/10), y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0, 255), 2)
        cv2.putText(frame, status, (100,150), font, 3 , (0,0, 255), 2, cv2.LINE_4)
        cv2.rectangle(frame,(x,y), (x+w, y+h), (0,0,255))
        
    elif ((Predictions)==3):
        status = "Happy"
        x1,y1,w1,h1 = 0,0,175,75
        #black background
        cv2.rectangle(frame, (x1,x1), (x1+w1, y1+h1), (0,0,0), -1)
        #text
        cv2.putText(frame, status, (x1+int(w1/10), y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0, 255), 2)
        cv2.putText(frame, status, (100,150), font, 3 , (0,0, 255), 2, cv2.LINE_4)
        cv2.rectangle(frame,(x,y), (x+w, y+h), (0,0,255))
        
    elif ((Predictions)==4):
        status = "Sad"
        x1,y1,w1,h1 = 0,0,175,75
        #black background
        cv2.rectangle(frame, (x1,x1), (x1+w1, y1+h1), (0,0,0), -1)
        #text
        cv2.putText(frame, status, (x1+int(w1/10), y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0, 255), 2)
        cv2.putText(frame, status, (100,150), font, 3 , (0,0, 255), 2, cv2.LINE_4)
        cv2.rectangle(frame,(x,y), (x+w, y+h), (0,0,255))
        
    elif ((Predictions)==5):
        status = "Surprise"
        x1,y1,w1,h1 = 0,0,175,75
        #black background
        cv2.rectangle(frame, (x1,x1), (x1+w1, y1+h1), (0,0,0), -1)
        #text
        cv2.putText(frame, status, (x1+int(w1/10), y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0, 255), 2)
        cv2.putText(frame, status, (100,150), font, 3 , (0,0, 255), 2, cv2.LINE_4)
        cv2.rectangle(frame,(x,y), (x+w, y+h), (0,0,255))
        
    elif ((Predictions)==6):
        status = "Neutral"
        x1,y1,w1,h1 = 0,0,175,75
        #black background
        cv2.rectangle(frame, (x1,x1), (x1+w1, y1+h1), (0,0,0), -1)
        #text
        cv2.putText(frame, status, (x1+int(w1/10), y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0, 255), 2)
        cv2.putText(frame, status, (100,150), font, 3 , (0,0, 255), 2, cv2.LINE_4)
        cv2.rectangle(frame,(x,y), (x+w, y+h), (0,0,255))
        
        
    cv2.imshow('Face Emotion Recognition', frame)
    
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
