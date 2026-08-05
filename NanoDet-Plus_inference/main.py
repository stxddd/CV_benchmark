import cv2
import numpy as np
import onnxruntime as ort
import time


MODEL = "nanodet-plus-m-1.5x_320.onnx"


INPUT_SIZE = 320

CONF_THRESHOLD = 0.45
NMS_THRESHOLD = 0.5

REG_MAX = 7



COCO = [
"person","bicycle","car","motorcycle","airplane",
"bus","train","truck","boat","traffic light",
"fire hydrant","stop sign","parking meter","bench",
"bird","cat","dog","horse","sheep","cow",
"elephant","bear","zebra","giraffe","backpack",
"umbrella","handbag","tie","suitcase","frisbee",
"skis","snowboard","sports ball","kite",
"baseball bat","baseball glove","skateboard",
"surfboard","tennis racket","bottle",
"wine glass","cup","fork","knife","spoon",
"bowl","banana","apple","sandwich",
"orange","broccoli","carrot","hot dog",
"pizza","donut","cake","chair","couch",
"potted plant","bed","dining table",
"toilet","tv","laptop","mouse","remote",
"keyboard","cell phone","microwave",
"oven","toaster","sink","refrigerator",
"book","clock","vase","scissors",
"teddy bear","hair drier","toothbrush"
]


# ======================================
# LOAD MODEL
# ======================================

session = ort.InferenceSession(
    MODEL,
    providers=[
        "CPUExecutionProvider"
    ]
)


input_name = session.get_inputs()[0].name


print("INPUT:")
print(session.get_inputs()[0].shape)

print("OUTPUT:")
for o in session.get_outputs():
    print(o.name,o.shape)



# ======================================
# GRID
# ======================================

def create_points():

    points=[]

    for stride in [8,16,32,64]:

        size = INPUT_SIZE // stride


        for y in range(size):

            for x in range(size):

                points.append(
                    [
                        x * stride,
                        y * stride,
                        stride
                    ]
                )


    return np.array(
        points,
        dtype=np.float32
    )



POINTS=create_points()


print(
    "points:",
    POINTS.shape
)



# ======================================
# PREPROCESS
# ======================================

def preprocess(frame):

    img=cv2.resize(
        frame,
        (INPUT_SIZE,INPUT_SIZE)
    )


    img=cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )


    img=img.astype(
        np.float32
    )


    mean=np.array(
        [
            103.53,
            116.28,
            123.675
        ],
        dtype=np.float32
    )


    std=np.array(
        [
            57.375,
            57.12,
            58.395
        ],
        dtype=np.float32
    )


    img=(img-mean)/std


    img=img.transpose(
        2,0,1
    )


    img=np.expand_dims(
        img,
        0
    )


    return img.astype(
        np.float32
    )



# ======================================
# SOFTMAX
# ======================================

def softmax(x):

    x=x-np.max(
        x,
        axis=1,
        keepdims=True
    )

    e=np.exp(x)


    return e/e.sum(
        axis=1,
        keepdims=True
    )



# ======================================
# BBOX
# ======================================

def distance2bbox(points,dist):


    x=points[:,0]
    y=points[:,1]


    l=dist[:,0]
    t=dist[:,1]
    r=dist[:,2]
    b=dist[:,3]


    return np.stack(
        [
            x-l,
            y-t,
            x+r,
            y+b
        ],
        axis=1
    )



# ======================================
# DECODE
# ======================================

def decode(output,shape):


    pred=output[0]


    cls=pred[:,:80]

    reg=pred[:,80:]



    # IMPORTANT:
    # this ONNX already has sigmoid scores

    scores=cls


    labels=np.argmax(
        scores,
        axis=1
    )


    conf=np.max(
        scores,
        axis=1
    )



    mask=conf>CONF_THRESHOLD



    if mask.sum()==0:

        return [],[],[]



    conf=conf[mask]

    labels=labels[mask]

    reg=reg[mask]

    points=POINTS[mask]



    # DFL decode

    reg=reg.reshape(
        -1,
        4,
        REG_MAX+1
    )


    reg=softmax(
        reg.reshape(
            -1,
            REG_MAX+1
        )
    )


    proj=np.arange(
        REG_MAX+1,
        dtype=np.float32
    )


    reg=(reg*proj).sum(
        axis=1
    )


    reg=reg.reshape(
        -1,
        4
    )


    reg*=points[:,2,None]



    boxes=distance2bbox(
        points,
        reg
    )



    h,w=shape[:2]


    boxes[:,[0,2]]*=w/INPUT_SIZE

    boxes[:,[1,3]]*=h/INPUT_SIZE



    boxes[:,0]=np.clip(
        boxes[:,0],
        0,
        w
    )

    boxes[:,2]=np.clip(
        boxes[:,2],
        0,
        w
    )

    boxes[:,1]=np.clip(
        boxes[:,1],
        0,
        h
    )

    boxes[:,3]=np.clip(
        boxes[:,3],
        0,
        h
    )



    return (
        boxes,
        conf,
        labels
    )



# ======================================
# NMS
# ======================================

def nms(boxes,scores):


    rects=[]


    for b in boxes:

        x1,y1,x2,y2=b


        rects.append(
            [
                int(x1),
                int(y1),
                int(x2-x1),
                int(y2-y1)
            ]
        )



    indexes=cv2.dnn.NMSBoxes(
        rects,
        scores.tolist(),
        CONF_THRESHOLD,
        NMS_THRESHOLD
    )


    result=[]


    for i in indexes:

        if isinstance(
            i,
            np.ndarray
        ):
            result.append(
                int(i[0])
            )

        else:
            result.append(
                int(i)
            )


    return result



# ======================================
# CAMERA
# ======================================

cap=cv2.VideoCapture(0)


prev=time.time()


while True:


    ret,frame=cap.read()


    if not ret:
        break



    inp=preprocess(frame)



    output=session.run(
        None,
        {
            input_name:inp
        }
    )



    boxes,scores,labels=decode(
        output[0],
        frame.shape
    )



    if len(boxes):


        keep=nms(
            boxes,
            scores
        )


        for i in keep:


            x1,y1,x2,y2=boxes[i].astype(int)


            cv2.rectangle(
                frame,
                (x1,y1),
                (x2,y2),
                (0,255,0),
                2
            )


            cv2.putText(
                frame,
                f"{COCO[labels[i]]} {scores[i]:.2f}",
                (x1,y1-5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,255,0),
                2
            )



    fps=1/(time.time()-prev)

    prev=time.time()



    cv2.putText(
        frame,
        f"FPS {fps:.1f}",
        (20,30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,0,255),
        2
    )


    cv2.imshow(
        "NanoDet Plus ONNX",
        frame
    )


    if cv2.waitKey(1)==27:
        break



cap.release()
cv2.destroyAllWindows()