#ifndef CRNN2_H
#define CRNN2_H
#include "layer.h"
#include "net.h"
#include <algorithm>
#if defined(USE_NCNN_SIMPLEOCV)
#include "simpleocv.h"
#else
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#endif
#include <stdio.h>
#include <vector>
#include <numeric>
#include <fstream>
#include <string>
#include <iostream>
#include <exception>
#include <time.h>
#include "Net_Creator.h"
using namespace std;

class CRNN_Recognize:public Net_Creator
{
    public:
        CRNN_Recognize(string path,bool int8_flag):Net_Creator(path,int8_flag)
        {
          cout<<"crnn初始化"<<endl;
         
        }
         virtual void detect(const cv::Mat& bgr,string &rec);
};
#endif