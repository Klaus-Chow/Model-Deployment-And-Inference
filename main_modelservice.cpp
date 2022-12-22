#include <iostream>
#include "Utils_crnn.h"
#include "Crnn.h"
#include "Yolov5.h"
#include "Utils_yolo.h"
#include "ModelService.h"
#include <string>
#include <sys/timeb.h>
#include <map>
#include "Tools.h"
using namespace std;
int main(int argc, char** argv)
{
    
    ///读入目标图像
     if (argc != 2)
        {
            fprintf(stderr, "Usage: %s [imagepath]\n", argv[0]);
            return -1;
        }
        const char* imagepath = argv[1];
        string img_name=imagepath;

        cv::Mat m = cv::imread(imagepath, 1); //BGR
    //     cout<<m.cols<<" "<<m.rows<<endl;//w,h
        if (m.empty())
        {
            fprintf(stderr, "cv::imread %s failed\n", imagepath);
            return -1;
        }
    
   ///--------加载身份证检测模型,边缘检测模型，文本检测模型，文本识别模型 
    string yolo_detect_model_path,yolo_detect_edge_model_path,yolo_detect_text_model_path,crnn_rec_text_model_path;
    yolo_detect_model_path="/src/notebooks/ncnn-20220420/build/tools/quantize/best_lite-sim_int8.param";
    yolo_detect_edge_model_path="/src/notebooks/ncnn-20220420/build/tools/quantize/best-edge-int8.param";
    yolo_detect_text_model_path="/src/notebooks/ncnn-20220420/build/tools/quantize/best-text-int8.param";
    crnn_rec_text_model_path="/src/notebooks/ncnn-20220420/build/tools/quantize/id_crnn_mobile_fix_sim_int8.param";
    modelservice ms(yolo_detect_model_path,yolo_detect_edge_model_path,yolo_detect_text_model_path,crnn_rec_text_model_path);
    std::vector<Object> objects;
    std::vector<Object> edgeCoords;
    std::vector<Object> textbbox;
    std::vector<vector<float>>textbboxes;
    std::vector<int> nclasses;
    
    
    typedef map<string,int> Code;
    Code code;
    
    //身份证检测
    struct timeb startTime_sfz,endTime_sfz;
    ftime(&startTime_sfz);
    try
    {
      ms.yolo_detect.detect(m,objects);
      code["sfz_detect"]=200;  
    }
    catch(exception e)
    {
    
//         cout<<"身份证检测存在错误!"<<endl;
        code["sfz_detect"]=5001;
    
    }
    ftime(&endTime_sfz);
    cout<<"sfz检测时间"<<(endTime_sfz.time-startTime_sfz.time)*1000+(endTime_sfz.millitm-startTime_sfz.millitm)<<"ms"<<endl;
// //     画出bbox
//     draw_objects(m, objects,img_name);
//     //扣除ROI
//     cv::Mat ROI;
//     crop_ROI(m,ROI,objects,img_name);
    
    
    cv::Mat im_c,imshow,im_p,im_r;
    bool perspective;
    string ID_type;
    int flag=1;
    
    if(!objects.size())
    {
    
       cout<<"未检测到sfz"<<endl; 
    }
    else //检测到sfz
    {
        struct timeb startTime_edge,endTime_edge;
        for(int i=0;i<objects.size();i++) //遍历每一个目标
        {
            vector<float> result_conf;
            vector<string> recog_result;
            const Object& obj=objects[i];
            GetCropImageAndIDType(im_c,ID_type,m,obj,true);//截取目标的ROI得到目标im_c,并且返回当前类别
//             cout<<im_c.cols<<" "<<im_c.rows<<endl; //w.h
            im_c.copyTo(imshow);
//             cv::imwrite("../Output/im_c.jpg",imshow);
            
            //边缘检测
            ftime(&startTime_edge);
            try
            {
                ms.yolo_detect_edge.detect(im_c,edgeCoords);
                code["edge_detect"]=201;  
                ftime(&endTime_edge);
            
            }
            catch(exception e)
            {
            //         cout<<"边缘检测存在错误!"<<endl;
                code["edge_detect"]=5002;
            
            
            }
           cout<<"edge检测时间"<<(endTime_edge.time-startTime_edge.time)*1000+(endTime_edge.millitm-startTime_edge.millitm)<<"ms"<<endl;
//            cout<<"num of edge_boxes: "<<edgeCoords.size()<<endl;
//            draw_objects(im_c, edgeCoords,img_name.replace(img_name.find(".jpg"),img_name.find(".jpg")+3,"_edge.jpg"));
           
           if(edgeCoords.size()>0)
           {
               GetImgByEdge(im_p,ID_type,im_c,edgeCoords,perspective);//边缘检测后进行透视变换
                if(perspective && !(im_p.empty()))
                {
                    im_p.copyTo(im_r);
//                     cout<<"保存边缘检测后仿射变换的图像"<<endl;
//                     cv::imwrite("../Output/im_p.jpg",im_r);
                }
               else
               {
               
               
                    GetCropImageAndIDType(im_p,ID_type,m,obj,false);

    //                 SIFT变换
                    bool get_perspect = false;
                    im_r=getPerImgInF(im_p,ID_type,get_perspect);  
//                     cv::imwrite("../Output/sift_im_r.jpg",im_r);
               
               }
           }
           else
           {
           
                   GetCropImageAndIDType(im_r,ID_type,m,obj,false);
                    //SIFT变换
                   bool get_perspect = false;
                   im_r=getPerImgInF(im_r,ID_type,get_perspect); 
//                    cv::imwrite("../Output/sift_im_r.jpg",im_r);
           
           }
            
            /////////Text detect//////////////////////
           struct timeb startTime_text,endTime_text;
           try
           {
               ftime(&startTime_text);
               ms.yolo_detect_text.detect(im_r,textbbox);
               ftime(&endTime_text);
               code["text_detect"]=202;
           }
           catch(exception e)
           {
           
               code["text_detect"]=5003;//文本检测失败
           
           }
           cout<<"text检测时间"<<(endTime_text.time-startTime_text.time)*1000+(endTime_text.millitm-startTime_text.millitm)<<"ms"<<endl;
           fixbox(textbboxes,nclasses,textbbox); //将xywh转换成x1y1x2y1x2y2x1y2格式
           
           
            
            
//            ms.crnn_recognize.log_crnn();
          
           string result_rec;
           for(int i=0;i<textbboxes.size();i++) //遍历所有的textbox,扣除ROI并识别
           {
               cv::Mat img_c_text,im_c_text_p;
               vector<float> coord;
               coord.push_back(textbboxes[i][0]);//x1
               coord.push_back(textbboxes[i][1]);//y1
               coord.push_back(textbboxes[i][2]);//x2
               coord.push_back(textbboxes[i][3]);//y1
               coord.push_back(textbboxes[i][4]);//x2
               coord.push_back(textbboxes[i][5]);//y2
               coord.push_back(textbboxes[i][6]);//x1
               coord.push_back(textbboxes[i][7]);//y2      
           
               cv::Mat coord_point=cv::Mat(coord,true);//8*1
               im_c_text_p=get_perspective_image_rec_result(im_r,coord_point);//图像扣取校正，从透视变换后的图像上扣取文本bbox并且也进行透视变换校正                 
//                //在c++张遍历保留不同名字的方式
//               std::ostringstream name;
//               name<<"../Output/text_detect_pers_"<<i<<".jpg";
//               cv::imwrite(name.str(),im_c_text_p);

               
               //////////////////CRNN rec//////////////////////
               //将扣出来的ROI传入crnn识别
               try
               {
                   result_rec=ms.crnn_recognize.detect_crnn(im_c_text_p);
                   code["crnn_rec"]=203; ///识别成功 
               
               }
               catch(exception e)
               {
                   code["crr_rec"]=5003;//文本识别失败
               
               }
               recog_result.push_back(result_rec);
               
           }
           if((int)(textbboxes.size())<6 && ID_type=="obverse")
            {
                cout<<"类别出错，应该属于背面"<<endl;
                ID_type="reverse";
            }
            else if((int)(textbboxes.size())>6 && ID_type=="reverse")
            {
                cout<<"类别出错，应该属于正面"<<endl;
                ID_type="obverse";
            }
            struct ID_Result_Obverse Obverse_result; 
            struct ID_Result_Reverse Reverse_result;
            
            
            //后处理，正面识别校正
            if(ID_type=="obverse")
            {
                get_result_observe(&Obverse_result,nclasses,textbboxes,recog_result);
            }
            else
            {
                get_result_reserve(&Reverse_result,nclasses,textbboxes,recog_result);
            }
            
            typedef map<string,string> Recog_Item;
            Recog_Item recog_item;
            if(ID_type=="obverse")
            {
                recog_item["name"]=Obverse_result.name;
                recog_item["sex"]=Obverse_result.sex;
                recog_item["national"]=Obverse_result.national;
                recog_item["birth"]=Obverse_result.birth;
                recog_item["address"]=Obverse_result.address;
                recog_item["id_number"]=Obverse_result.id_number;
                recog_item["id_type"]=ID_type;
                recog_item["issued"]="";
                recog_item["effective_date"]="";
            }
            else
            {
                recog_item["name"]="";
                recog_item["sex"]="";
                recog_item["national"]="";
                recog_item["birth"]="";
                recog_item["address"]="";
                recog_item["id_number"]="";
                recog_item["issued"]=Reverse_result.issued;
                recog_item["effective_date"]=Reverse_result.effective_date;
                recog_item["id_type"]=ID_type;
            }
            cout<<recog_item["name"]<<" "<<recog_item["sex"]<<" "
                <<recog_item["national"]<<" "<<recog_item["birth"]<<" "
                <<recog_item["address"]<<" "<<recog_item["id_number"]<<" "
                <<recog_item["issued"]<<" "<<recog_item["effective_date"]<<" "<<recog_item["id_type"]<<endl;
            
        }
    

    }
    return 0;
    
    
}