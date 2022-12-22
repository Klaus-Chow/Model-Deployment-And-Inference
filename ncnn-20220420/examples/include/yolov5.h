#ifndef _YOLOV5_H
#define _YOLOV5_H
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <string>
#include <vector>
#include "layer.h"
#include "net.h"
#include "simpleocv.h"
#include <iostream>
struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};
static int detect_yolov5(const cv::Mat& bgr, std::string& model_param_path,std::vector<Object>& objects);
static void generate_proposals(const ncnn::Mat& anchors, 
                               int stride,
                               const ncnn::Mat& in_pad, 
                               const ncnn::Mat& feat_blob,
                               float prob_threshold, 
                               std::vector<Object>& objects
                               )
#endif