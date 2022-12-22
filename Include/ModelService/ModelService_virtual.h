#ifndef MODELSERVICE_VIRTUAL_H
#define MODELSERVICE_VIRTUAL_H
#include <iostream>
// #include "Utils_crnn.h"
// #include "Crnn2.h"
#include "yolov5_virtual.h"
#include "Utils_yolo.h"
#include <string>
using namespace std;
struct config_opt
{
    string model_path;
    bool int8_flag;
    float prob_threshold;
    float nms_threshold;
};
class ModelService
{
   public:
        YoloInfer *yolo_detect_edge_p,*yolo_detect_p,*yolo_detect_text_p;
        ModelService(vector<struct config_opt>config_opt_modelservice)
        {
            cout<<"---------load modelservice!-----------"<<endl;
             yolo_detect_p=new YoloInfer(config_opt_modelservice[0].model_path,config_opt_modelservice[0].int8_flag,config_opt_modelservice[0].prob_threshold,config_opt_modelservice[0].nms_threshold);
            yolo_detect_edge_p=new YoloInfer(config_opt_modelservice[1].model_path,config_opt_modelservice[1].int8_flag,config_opt_modelservice[1].prob_threshold,config_opt_modelservice[1].nms_threshold);
            yolo_detect_text_p=new YoloInfer(config_opt_modelservice[2].model_path,config_opt_modelservice[2].int8_flag,config_opt_modelservice[2].prob_threshold,config_opt_modelservice[2].nms_threshold);        
        }
        ModelService(struct config_opt yd,struct config_opt ydep,struct config_opt ydtp):yd_p(yd),yde_p(ydep),ydt_p(ydtp)//,config_opt yde,config_opt ydt,config_opt crt
        {
            cout<<"---------load modelservice!-----------"<<endl;
            yolo_detect_p=new YoloInfer(yd_p.model_path,yd_p.int8_flag,yd_p.prob_threshold,yd_p.nms_threshold);
            yolo_detect_edge_p=new YoloInfer(yde_p.model_path,yde_p.int8_flag,yde_p.prob_threshold,yde_p.nms_threshold);
            yolo_detect_text_p=new YoloInfer(ydt_p.model_path,ydt_p.int8_flag,ydt_p.prob_threshold,ydt_p.nms_threshold);
        }
        
   private:
       struct config_opt yd_p;
       struct config_opt yde_p; 
       struct config_opt ydt_p;
};
#endif