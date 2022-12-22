#include <iostream>
#include "ModelService_virtual.h"
#include <string>
#include <sys/timeb.h>
#include <map>
// #include "Tools.h"
#include "Net_Detect.h"
#include <fstream>
#include <unistd.h>
using namespace std;

//将model_service_opt_config.txt中的string按照空格存放到一个数组中
 vector<string> string2vetcor(string s1)
{
    /*
    s1:string 比如"/src/notebooks/c++_ID_Img_Rec_Project/onnx2ncnn_model/yolo_mtz/best-mtz_sim_int8.param" true 0.25 0.45
    */
    
    vector<string>s1_vector;
    string substring=" ";
    int index=0;
    int loc_blank;
    int s_length=s1.length(); 
    while(index<s_length)
    {
        
        loc_blank=s1.find(substring,index);
        if(loc_blank<0)
        {
            s1_vector.push_back(s1.substr(index,s_length));
            break;
        
        }
        s1_vector.push_back(s1.substr(index,loc_blank-index));
        index=loc_blank+1;
    
    }
    
    return s1_vector;
}



int main(int argc, char** argv)
{
    
    
    //---------------------------------------------------加载输入图像
    if(argc!=2)
    {
        fprintf(stderr,"Usage: %s [imagepath]\n",argv[0]);
        return -1;
    }
    const char* imagepath=argv[1];
    string img_name=imagepath;
    
    cv::Mat m=cv::imread(imagepath,1); //BGR
    if(m.empty())
    {
        fprintf(stderr,"cv::imread %s failed\n",imagepath);
        return -1;
    }
    std::vector<Object> objects;
    
    
    
    //-----------------------------------------------------配置文件
    /*
    1.yolo模型的路径、int8_flag、probthresh、nms_thresh
    2.如果是yolo模型还需要给出class_name;
    */
    ifstream inf;
    char buffer[256];//存放当前路径
    getcwd(buffer,sizeof(buffer));
    string root_path=buffer;//xxxxxxxxxxxxxxxx/build
    
    
    int loc=root_path.rfind('/');
    string root_path_parent=root_path.substr(0,loc);//xxxxxxxxxxxxxxxxxxxx/
    
    
    inf.open(root_path_parent+"//"+"model_service_opt_config.txt");//只能使用绝对路径
    string s;
    vector<string>s_vector;

    vector<struct config_opt>config_opt_modelservice;

    while(getline(inf,s))
    {
        struct config_opt co;
        s_vector=string2vetcor(s);
        co.model_path = s_vector[0];
        istringstream(s_vector[1])>>boolalpha>>co.int8_flag;//string：“true”,“false” 转bool,string " 1"转bool的话不加boolalpha
        co.prob_threshold=stof(s_vector[2]); //string转float:stof,转int:stoi
        co.nms_threshold=stof(s_vector[3]);        
        config_opt_modelservice.push_back(co); 
    }
      
    
    
        
    //----------------------------------------------------------加载ModelService
    ModelService ms(config_opt_modelservice);
//     ModelService ms(config_opt_modelservice[0],config_opt_modelservice[1],config_opt_modelservice[2]);
    
    Net_detect(ms.yolo_detect_p,m,objects);
//     Net_detect(ms.yolo_detect_edge_p,m,objects); //mtz检测、
//     Net_detect(ms.yolo_detect_text_p,m,objects);
    cout<<objects.size()<<endl;
    cout<<img_name<<endl;
    
    
    
    //画出bbox
    draw_objects(m,objects,img_name);
    
//     //扣除ROI
//     cv::Mat ROI;
//     crop_ROI(m,ROI,objects,img_name);
    
    
    cout<<"我太帅楼！"<<endl;
    return 0;
}