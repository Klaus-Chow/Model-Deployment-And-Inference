#include "Crnn2.h"
#include"Utils_crnn.h"
using namespace std;

void CRNN_Recognize::detect(const cv::Mat& bgr,string &rec)
{

    const int dstHeight=32;
    const int dstWidth=280;
    
    int hi=bgr.rows;
    int wid=bgr.cols;
    
    float scale=32.0/(float)(hi);
    int wid_new=(int)(scale*wid);
    
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_RGB, bgr.cols, bgr.rows,wid_new, dstHeight);
    const float meanValues[3]={127.5,127.5,127.5};
    const float normValues[3]={1.0/127.5,1.0/127.5,1.0/127.5};
    
    in.substract_mean_normalize(meanValues,normValues);
    
    ncnn::Extractor ex=net.create_extractor();
    ex.input("input",in);
    
    ncnn::Mat out;
    ex.extract("output",out);
    
    //后处理
    rec= scoreToTextLine((float *) out.data, out.h, out.w);
}