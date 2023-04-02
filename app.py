from ultralytics import YOLO
import cv2
import cvzone
import math
import streamlit as st

with st.sidebar : 
    st.image("2.png" , width=300)
    select_type_detect = st.selectbox("Detection from :  ",
                                            ("File", 
                                             "Live"))
    select_device = st.selectbox("Select compute Device :  ",
                                            ("CPU", "GPU"))
    save_output_video = st.radio("Save output video?",
                                            ('Yes', 'No'))

    confd = st.slider("Select threshold confidence value : " , min_value=0.1 , max_value=1.0 , value=0.25)
    iou = st.slider("Select Intersection over union (iou) value : " , min_value=0.1 , max_value=1.0 , value=0.5)

tab0 , tab1 = st.tabs(["Home" , "Detection"])
with tab0:
    st.header("About MY Project : ")
    st.image("smokey.gif")
    st.write("""The Wildfire Smoke Detection and Tracking System is a high-tech solution that uses drones or HPWREN cameras equipped with the YOLOv8 algorithm to detect and alert individuals of potential wildfire smoke in the surrounding area. 
    The system is designed to work in real-time, constantly monitoring the environment and providing accurate and timely notifications of any smoke detected. The use of drones allows for a wider coverage area and the ability to access hard-to-reach areas while HPWREN cameras are utilized to monitor remote areas that are at high risk of wildfire.
    The YOLOv8 algorithm is a powerful machine learning tool that is able to detect and classify objects in real-time, making it a valuable tool in detecting smoke and preventing the spread of wildfire.""")
    st.header("About Dataset : ")
    st.write("This dataset is released by AI for Mankind in collaboration with HPWREN under a Creative Commons by Attribution Non-Commercial Share Alike license. The original dataset (and additional images without bounding boxes) can be found in their GitHub repo.")
    st.write("https://github.com/aiformankind/wildfire-smoke-dataset")


with tab1 : 
    if select_device == "GPU" : 
        DEVICE_NAME = st.selectbox("Select GPU index : " , 
                                     (0, 1 , 2)) 
    elif select_device =="CPU" : 
        DEVICE_NAME = "cpu"
    fpsReader = cvzone.FPS()
    class_names = ["Smoke"]
    if select_type_detect == "File" : 
        file = st.file_uploader("Select Your File : " ,
                                 type=["mp4" , "mkv"])
        if file : 
            source = file.name
            cap = cv2.VideoCapture(source)
    elif select_type_detect == "Live" : 
        source = st.text_input("Past Your Url here and Click Entre")
        cap = cv2.VideoCapture(source)
    # creat the model
    model = YOLO("wildfirev2.pt")
    frame_window = st.image( [] )
    start , stop = st.columns(2)
    with start : 
        start = st.button("Click To Start")
    with stop : 
        stop = st.button("Click To Stop" , key="ghedqLKHF")
    if start :
        while True :
            _ , img = cap.read() 
            if save_output_video == "Yes" :
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
                fourcc = cv2.VideoWriter_fourcc(*'MP4V') #use any fourcc type to improve quality for the saved video
                out = cv2.VideoWriter(f'results/{source.split(".")[0]}.mp4', fourcc, 10, (w, h)) #Video settings
            # fps counter
            fps, img = fpsReader.update(img,pos=(20,50),
                                        color=(0,255,0),
                                        scale=2,thickness=3)
            # make the prediction 
            results = model(img ,conf=confd ,
                             iou=iou,
                             device=DEVICE_NAME)
            for result in results : 
                # depackage results
                bboxs = result.boxes 
                for box in bboxs : 
                    # bboxes
                    x1  , y1 , x2 , y2 = box.xyxy[0]
                    x1  , y1 , x2 , y2 = int(x1)  , int(y1) , int(x2) , int(y2)
                    # confidence 
                    conf = math.ceil((box.conf[0] * 100 ))
                    # class name
                    clsi = int(box.cls[0])
                    # calculate the width and the height
                    w,h = x2 - x1 , y2 - y1
                    # convert it into int
                    w , h = int(w) , int(h)
                    # draw our bboxes 
                    cvzone.cornerRect(img , 
                                    (x1 , y1 , w , h) ,
                                        l=7)
                    # put information inside our image
                    cvzone.putTextRect(img , f"{conf} % {class_names[clsi]}" , 
                                        (max(0,x1) , max(20 , y1)),
                                        thickness=1 ,
                                        colorR=(0,0,255) , 
                                        scale=0.9 , offset=3)
                    try: 
                        out.write(img)
                    except:
                        pass

            frame  = cv2.cvtColor( img , 
                                cv2.COLOR_BGR2RGB)
            frame_window.image(frame)
    else:
        try:
            cap.release()
        except : 
            pass
