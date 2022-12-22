#ifndef CRNN_H
#define CRNN_H
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
using namespace std;

class CRNN_Recognize
{
    public:
        CRNN_Recognize(string path):model_path_param(path)
        {
          cout<<"crnn初始化"<<endl;
          model_path_bin=path.replace(path.rfind("."),6,".bin");
          
          
          const char *param_path=model_path_param.c_str();
          const char *bin_path=model_path_bin.c_str();
          crnn.opt.num_threads=8;
          crnn.opt.use_int8_inference=true;
    
          crnn.load_param(param_path);
          crnn.load_model(bin_path);
          cout<<model_path_bin<<endl;
          
        }
        string detect_crnn(cv::Mat& bgr);
        void log_crnn()
        {
            cout<<"---使用crnn进行推理---"<<endl;
            cout<<"---加载模型权重:"<<model_path_param;
            cout<<model_path_bin<<endl;
        }
        
    private:
        string model_path_param,model_path_bin;
        ncnn::Net crnn;
        
    
};
#endif