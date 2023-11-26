import cv2
import requests
import time
import base64
from datetime import datetime


classNames = []
classFile = "/home/yellcw/Object_Detection_Files/coco.names"
with open(classFile,"rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

configPath = "/home/yellcw/Object_Detection_Files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "/home/yellcw/Object_Detection_Files/frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

last_sent_time= 0

def get_current_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def send_post_request(data):
    url = 'https://a67c-129-100-255-61.ngrok-free.app'
    response = requests.post(url,json=data)

def getObjects(img, thres, nms, draw=True, objects=[]):
    global last_sent_time
   
    classIds, confs, bbox = net.detect(img,confThreshold=thres,nmsThreshold=nms)
    #print(classIds,bbox)
    if len(objects) == 0: objects = classNames
    objectInfo =[]
    if len(classIds) != 0:
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            className = classNames[classId - 1]
            if className in objects:
                objectInfo.append([box,className])
                print("detect class")
                if (draw):
                    print("drawing...")
                    cv2.rectangle(img,box,color=(0,255,0),thickness=2)
                    cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
                    cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                    cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
                    cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                   
                    current_time = time.time()
                    if current_time - last_sent_time >= 0:
                        cv2.imwrite(get_current_time() + ".png", img)
                        print("image saved")
                       
                        post_data = {
                            'time': get_current_time(),
                            'className': classNames[classId - 1],
                            'confidence': round(confidence*100,2),
                            'box': box.tolist(),
                            'location':{'lat':43.00969,'long':-81.27273},
                            'image':base64.b64encode(cv2.imencode('.jpg',cv2.resize(img,(480,360)))[1]).decode('utf-8'),
                        }
                       
                        send_post_request(post_data)
                        print("data sent")
                        last_sent_time = current_time
                       
   
    return img,objectInfo


if __name__ == "__main__":

    cap = cv2.VideoCapture(0)
    cap.set(3,640)
    cap.set(4,480)
    #cap.set(10,70)
   
   
    while True:
        print("Program running")
        try:
            success, img = cap.read()
            if not success:
                print("Failed to capture frame")
                break
            result, objectInfo = getObjects(img,0.55,0.2, objects=['person'])
            #print(objectInfo)
            cv2.imshow("Output",img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
       
        except KeyboardInterrrupt:
            break