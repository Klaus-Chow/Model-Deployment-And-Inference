#include <iostream>
#include "Utils_crnn.h"
#include "Crnn.h"
#include "Yolov5.h"
#include "Utils_yolo.h"
#include <string>
using namespace std;
class modelservice
{
   public:
        modelservice(string ydm,string yde,string ydt,string crt):
        yolo_detect(ydm,0.5),yolo_detect_edge(yde,0.25),yolo_detect_text(ydt,0.5),crnn_recognize(crt)
        {
            cout<<"---------load modelservice!-----------"<<endl;
        }
        yoloInfer yolo_detect,yolo_detect_edge,yolo_detect_text;//yoloInfer定义的对象作为成员变量
        CRNN_Recognize crnn_recognize;
};