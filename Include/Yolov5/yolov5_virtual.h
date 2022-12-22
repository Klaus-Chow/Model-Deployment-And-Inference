#ifndef YOLOV5_VIRTUAL_H
#define YOLOV5_VIRTUAL_H
#include<iostream>
#include<string>
#include "layer.h"
#include "net.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <float.h>
#include <stdio.h>
#include <vector>
#include "Net_Creator.h"
using namespace std;
class YoloInfer:public Net_Creator
{
    public:
        YoloInfer(string path,bool int8_flag,float prob_th,float nms_th):Net_Creator(path,int8_flag),prob_threshold(prob_th),nms_threshold(nms_th)
        {
            cout<<"yolov5初始化"<<endl;
        }
        void detect(const cv::Mat& bgr,vector<Object>& objects);
    private:
        float prob_threshold;
        float nms_threshold;
            
};
#endif