#ifndef TOOL_H
#define TOOL_H
#include"Yolov5.h"
#include<iostream>
#include<string>
#include "layer.h"
#include "net.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include "opencv2/calib3d/calib3d.hpp"
#include <float.h>
#include <stdio.h>
#include <vector>
#include <map>
#include <regex>
//根据检测结果扣除图像并且做扩充，而且根据label类型做出角度校正（90,180,270,360），这个是对整张图像进行校正
int GetCropImageAndIDType(cv::Mat& im_crop,std::string& ID_type,const cv::Mat im_ori,const Object& obj,const bool padding=true);


//边缘检测模型，根据得到的点的类型对直线拟合校正图像中的sfz目标
int GetImgByEdge(cv::Mat& im_p,std::string& ID_type,const cv::Mat im_c,std::vector<Object>& edgeCoords,bool& perspective);

//透视变换
cv::Mat get_perspective_image(cv::Mat im_c,cv::Mat& coord);

//根据边缘检测的点进行拟合直线
float* linear_regression(vector<vector<float>>& line,bool &get_kb);

//找出两个相交线的交点
std::vector<float> crosspoint(float *k1_b1,float *k2_b2,float x1,float x3,bool target_r1,bool target_r2,bool& point_is_exist);

    
// 有效期限 template_arr[0];签发机关[1];居民身份证[2];中华人民共和国[3]
bool JudgeImgPSuccessF(const cv::Mat &template_img, const cv::Mat &result);

// 年 template_arr[0];月[1];日[2];姓名[3];性别[3];民族[3];出生[3];住址[3];身份证号码[3]
bool JudgeImgPSuccessZ(const cv::Mat &template_img, const cv::Mat &result, int numthresh=5, int meanthresh=15);

//sift变换
cv::Mat imresize(const cv::Mat &img, int img_max_size);
cv::Mat SIFTInF(const cv::Mat &inpic_ori, const cv::Mat &template_img,float ratio,string img_type);
cv::Mat getPerImgInF(const cv::Mat &inpic,string img_type,bool &get_perspect);


//将yolo预测的坐标x,y,w,h转换成文本识别的四个点（1*8），因为是yolo做文本检测所以输出需要改变
int fixbox(std::vector<vector<float>>& textbboxes,std::vector<int>& nclasses,std::vector<Object>& textbbox);
    
//图像扣取校正，从透视变换后的图像上扣取文本bbox并且也进行透视变换校正
cv::Mat get_perspective_image_rec_result(cv::Mat im_c,cv::Mat& coord);

struct ID_Result_Obverse
{
     //成员列表
    string name="";
    vector<float>name_coord={0,0,0,0,0,0,0,0};
    
    string national="";
    vector<float>national_coord={0,0,0,0,0,0,0,0};
    
    string sex="";
    vector<float>sex_coord={0,0,0,0,0,0,0,0};
    
    string year="";
    vector<float>year_coord={0,0,0,0,0,0,0,0};
    
    string month="";
    vector<float>month_coord={0,0,0,0,0,0,0,0};

    string day="";
    vector<float>day_coord={0,0,0,0,0,0,0,0};
    
    string birth="";
    vector<float>birth_coord={0,0,0,0,0,0,0,0};
    
    string address="";
    vector<float>address_coord={0,0,0,0,0,0,0,0};
    
    string id_number="";
    vector<float>id_number_coord={0,0,0,0,0,0,0,0}; 
    
    string id_type="obverse";
};
struct ID_Result_Reverse
{
     //成员列表
    string issued="";
    vector<float>issued_coord={0,0,0,0,0,0,0,0};
    
    string effective_date="";
    vector<float>effective_date_coord={0,0,0,0,0,0,0,0};
    
    string id_type="reverse";
};    
//对正面的识别结果进行后处理修正。
void get_result_observe(ID_Result_Obverse *Obverse_result,vector<int>&nclasses,vector<vector<float> >& bbox,vector<string>&reg_result);
//对反面的识别结果进行后处理修正
void get_result_reserve(ID_Result_Reverse *Reverse_result,vector<int>&nclasses,vector<vector<float> >& bbox,vector<string>&reg_result);
//计算距离，在透视变换的时候得到最大的矩形框
float cal_distance(float x1,float y1,float x2,float y2);
std::pair<float,float> cal_width_height(cv::Mat coord);
void nclassindex(vector<int>& vec,vector<int>& res,int nclass);
string matchString(string pattern_get,string content);
void get_similar_sex(map<string,vector<string>>& similar_sex);
void get_similar(map<string,vector<string>>& similar_data_nation,vector<string>&nationality);
string judge_nationality(string data_n);
string fix_sex(string sex_str);
template <typename T>
std::vector<size_t> sort_indexes(const std::vector<T> &v);
vector<string> split_utf8(string str);
string vecTostring(const vector<string> vec,size_t start,size_t end);
bool search(const vector<string> strs, string target);
#endif