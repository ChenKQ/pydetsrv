import requests
import sys, os
import base64
import time
import cv2


serverip = "172.20.10.3"
port = 8080
router = "json"
url = "http://" + serverip + ":" + port + "/" + router

img_root = 'D:\\projects\\BOX.AI\\项目设计\\电池检测\\data'
img_dir = ['difficult','easy']

def request(imgFile):
    with open(imgFile, 'rb') as f:
        rdata = f.read()
    e64data = base64.b64encode(rdata)
    prm = {'img': e64data}
    # ret = requests.get(url, params=prm)
    ret = requests.post(url, prm)
    print(ret.text)

def request2():
    prm = {'img': "hello json"}
    ret = requests.post(url, prm)
    print(ret.text)

def requestForAll():
    for fld in img_dir:
        imgs = os.listdir(os.path.join(img_root, fld))
        for img in imgs:  
            if img.endswith('.jpg') or img.endswith('JPG'):
                imgfile = os.path.join(img_root, fld, img)
                request(imgfile)

def TimeConnect():
    router = ""
    url = "http://" + serverip + ":" + port + "/" + router    
    for i in range(100):
        ret = requests.get(url)
        print(ret.text)

def TimeReceive():
    router = "receive"
    url = "http://" + serverip + ":" + port + "/" + router
    imgFile = "../test4.jpg"
    with open(imgFile, 'rb') as f:
        rdata = f.read()
    for i in range(100):
        # with open(imgFile, 'rb') as f:
        #     rdata = f.read()
        ret = requests.post(url, rdata)
        print(ret.text, len(rdata))

def TimeBase64Decode():
    router = "base64Decode"
    url = "http://" + serverip + ":" + port + "/" + router
    imgFile = "../test4.jpg"
    with open(imgFile, 'rb') as f:
        rdata = f.read()
        e64data = base64.b64encode(rdata)
        prm = {'img': e64data}
    for i in range(100):
        ret = requests.post(url, prm)
        print(ret.text, len(rdata))

def TimeReadImageOpenCV():
    router = "readImageOpenCV"
    url = "http://" + serverip + ":" + port + "/" + router
    imgFile = "../test4.jpg"
    with open(imgFile, 'rb') as f:
        rdata = f.read()
        e64data = base64.b64encode(rdata)
    prm = {'img': e64data}
    for i in range(100):
        ret = requests.post(url, prm)
        print(ret.text, len(rdata))

def main():
    tick = time.time()
    TimeReadImageOpenCV()
    tock  = time.time()
    print(tock-tick)
    # request("../send.jpg")
    # requestForAll()

if __name__ == '__main__':
    request2()
    # main()
    # imgFile = '../test.jpg'
    # img = cv2.imread(imgFile)
    # img2 = cv2.resize(img, (10,10))
    # cv2.imwrite('../test5.jpg', img2)