#ifndef UTILS_H
#define UTILS_H
// #include "Yolov5.h" //原始版本，不使用virtual相关
#include "Net_Creator.h" //virtual，需要注释第三行，否则struct Object会重复声明
#include <string.h>
inline float sigmoid(float x)
{
    return static_cast<float>(1.f / (1.f + exp(-x)));
} 
void generate_proposals(const ncnn::Mat& anchors, int stride, const ncnn::Mat& in_pad, const ncnn::Mat& feat_blob, float prob_threshold, std::vector<Object>& objects);//在grid上得到所有的预测的bbox(并且根据识别阈值（需要用归并排序）删除一部分阈值低的bboxes)
void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right);//对于候选框的识别score进行排序
void qsort_descent_inplace(std::vector<Object>& faceobjects);

//nms中IOU的面积交
inline float intersection_area(const Object& a, const Object& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold);//apply nms with nms_threshold
void draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects,string img_name);//在原图上绘制出检测出的目标
void crop_ROI(const cv::Mat& bgr,cv::Mat& ROI,const std::vector<Object>& objects,string img_name);//根据传入的坐标，扣除ROI部分
#endif
/*
主要是yolov5会用的一些函数，比如激活函数、生成候选框、对score排序(归并)、nms、IOU


*/