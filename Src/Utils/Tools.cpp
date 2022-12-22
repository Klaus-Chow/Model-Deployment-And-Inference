#include "Tools.h"
#include "Yolov5.h"
#include <map>
#include <numeric>
using namespace std;
// 有效期限 template_arr[0];签发机关[1];居民身份证[2];中华人民共和国[3]
bool JudgeImgPSuccessF(const cv::Mat &template_img, const cv::Mat &result){
    
    int template_arr[4][4] = {{197, 456, 313, 486},{197, 389, 313, 419},
                                     {255, 149, 779, 224},{283, 49, 745, 112}
                                    };
    for (int i = 0; i < 4; i++){
        template_arr[i][0] = (int) template_arr[i][0]/1.3375;
        template_arr[i][1] = (int) template_arr[i][1]/1.2981;
        template_arr[i][2] = (int) template_arr[i][2]/1.3375;
        template_arr[i][3] = (int) template_arr[i][3]/1.2981;      
    }

    int flag = 0;
    for (int j = 0; j < 4; j++){ 
        cv::Mat cut_template,cut_result;
        cut_template = template_img(cv::Range(template_arr[j][1],template_arr[j][3]),cv::Range(template_arr[j][0],template_arr[j][2])).clone(); 
        cut_result = result(cv::Range(template_arr[j][1],template_arr[j][3]),cv::Range(template_arr[j][0],template_arr[j][2])).clone(); 
        cv::cvtColor(cut_template, cut_template,cv::COLOR_RGB2GRAY);
        cv::cvtColor(cut_result, cut_result,cv::COLOR_RGB2GRAY);
        cv::threshold(cut_template,cut_template, 0, 255, cv::THRESH_OTSU);
        cv::threshold(cut_result,cut_result, 0, 255, cv::THRESH_OTSU);
        cut_template.convertTo(cut_template, CV_32F);
        cut_result.convertTo(cut_result, CV_32F);
        cut_template = cut_template - cut_result;
        cut_template.setTo(1,cut_template < 0);
        cut_template.convertTo(cut_template, CV_32F);
        cv::Scalar get_mean;
        get_mean = cv::mean(cut_template);
//         std::cout << "cut_template sum： " << cv::sum(cut_template) << std::endl;
//         std::cout << "cut_template mean： " << cv::mean(cut_template) << std::endl; 

        if (get_mean[0] < 15){
            flag += 1;
            } 
    }
    if (flag >= 2){
        return true;
        }
    return false;      
}


// 年 template_arr[0];月[1];日[2];姓名[3];性别[3];民族[3];出生[3];住址[3];身份证号码[3]
bool JudgeImgPSuccessZ(const cv::Mat &template_img, const cv::Mat &result, int numthresh, int meanthresh){

    
    int template_arr[9][4] = {{260,208,291,239},{337,208,368,239},
                              {416,208,447,239},{70,81,143,112},
                              {70,142,143,173},{259,144,332,175},
                              {68,208,141,239},{71,278,142,309},
                              {70,449,261,481}
                                    };
    for (int i = 0; i < 9; i++){
        template_arr[i][0] = (int) template_arr[i][0]/1.3375;
        template_arr[i][1] = (int) template_arr[i][1]/1.2981;
        template_arr[i][2] = (int) template_arr[i][2]/1.3375;
        template_arr[i][3] = (int) template_arr[i][3]/1.2981;      
    }

    int flag = 0;
    for (int j = 0; j < 9; j++){ 
        cv::Mat cut_template,cut_result;
        cut_template = template_img(cv::Range(template_arr[j][1],template_arr[j][3]),cv::Range(template_arr[j][0],template_arr[j][2])).clone(); 
        cut_result = result(cv::Range(template_arr[j][1],template_arr[j][3]),cv::Range(template_arr[j][0],template_arr[j][2])).clone(); 
        cv::cvtColor(cut_template, cut_template,cv::COLOR_RGB2GRAY);
        cv::cvtColor(cut_result, cut_result,cv::COLOR_RGB2GRAY);
        cv::threshold(cut_template,cut_template, 0, 255, cv::THRESH_OTSU);
        cv::threshold(cut_result,cut_result, 0, 255, cv::THRESH_OTSU);
        cut_template.convertTo(cut_template, CV_32F);
        cut_result.convertTo(cut_result, CV_32F);
        cut_template = cut_template - cut_result;
        cut_template.setTo(1,cut_template < 0);
        cv::Scalar get_mean;
        get_mean = cv::mean(cut_template);
//         std::cout << "cut_template sum： " << cv::sum(cut_template) << std::endl;
//         std::cout << "cut_template mean： " << cv::mean(cut_template) << std::endl; 
        if (get_mean[0] < meanthresh){
            flag += 1;
            } 
    }
    if (flag >= numthresh){
        return true;
        }
    return false;      
}




//找出两个直线的交点
std::vector<float> crosspoint(float *k1_b1,float *k2_b2,float x1,float x3,bool target_r1,bool target_r2,bool& point_is_exist)
{
    /*
    k1_b1：斜率和偏置
    k2_b2
    x1
    x3
    target_r1：
    target_r2：
    point_is_exist：是否存在交点
    
    */
    vector<float>point;
    point_is_exist=false;
    if(target_r1 == false)
    {
        if(target_r2)
        {
          float x=x1;
          float y=k2_b2[1]*x1+k2_b2[0];
          point.push_back(x);
          point.push_back(y);
          point_is_exist=true;
            
        }
    }
    else if(target_r2==false)
    {
        float x=x3;
        float y=k1_b1[1]*x3+k1_b1[0];
        point.push_back(x);
        point.push_back(y);
    }
    else if(k1_b1[1]!=k2_b2[1])
    {
        float x=(k2_b2[0]-k1_b1[0])/(k1_b1[1]-k2_b2[1]);
        float y=k1_b1[1]*x+k1_b1[0];
        point.push_back(x);
        point.push_back(y);
        point_is_exist=true;
    }
    return point;
}

template<typename T>
T SumVector(vector<T>& vec)
{
    T res = 0;
    for (size_t i=0; i<vec.size(); i++)
    {
        res += vec[i];
    }
    return res;
}

void linalgsolve(const float A[4],const float b[2],float result[2],bool &get_kb)
{
    try
    {
        result[1] = (A[0]*b[1] - A[2]*b[0])/((A[0]*A[3])-(A[2]*A[1]));
        result[0] = (b[0] - A[1]*result[1])/A[0];
        get_kb = true; 
        }
    catch(exception e){
        get_kb = false; 
    }  
}

float* linear_regression(vector<vector<float>>& line,bool &get_kb)
{
    float *result = new float[2];
    int N = line.size();
    vector<float>x(N);
    vector<float>x2(N);
    vector<float>y(N);
    vector<float>xy(N);
    float b[2];
    float A[4];
    for(int i=0;i<N;i++) 
    {
        x[i] = line[i][0];
        y[i] = line[i][1];
        x2[i] = x[i]*x[i];
        xy[i] = x[i]*y[i];
//         cout << "x: " << x2[i] << " ";
    }   
    b[0] = SumVector(y); //(float) accumulate(y.begin(), y.end(), 0);
    b[1] = SumVector(xy); //accumulate(xy.begin(), xy.end(), 0);
    A[0] = N;
    A[1] = SumVector(x); // accumulate(x.begin(), x.end(), 0);
    A[2] = SumVector(x); //accumulate(x.begin(), x.end(), 0);
    A[3] = SumVector(x2); //accumulate(x2.begin(), x2.end(), 0);

//     result[1] = (A[0]*b[1] - A[2]*b[0])/((A[0]*A[3])-(A[2]*A[1]));
//     result[0] = (b[0] - A[1]*result[1])/A[0];
    linalgsolve(A,b,result,get_kb);    
    return result;
}

cv::Mat get_perspective_image(cv::Mat im_c,cv::Mat& coord)
{
    cv::Mat im_p;
    float width,height;
    width=640.0;
    height=416.0;
    float a[4][2]={{0.0,0.0},{width,0.0},{width,height},{0.0,height}};
    cv::Mat pts1(4,2,CV_32FC1,a);
    cv::Mat pts2 = coord.reshape(2,4);
//     cout<<pts1<<endl;
//     cout<<pts2<<endl;
    cv::Mat H=cv::getPerspectiveTransform(pts2,pts1);
    cv::warpPerspective(im_c,im_p,H,cv::Size(640,416));
    return im_p;
}
int GetCropImageAndIDType(cv::Mat& im_crop,std::string& ID_type,const cv::Mat im_ori,const Object& obj,const bool padding)
{
    /*
    im_crop ：扣除的图像（ROI）
    ID_type : 根据
    im_ori :目标图像
    obj ：当前检测框的struct
    padding : bbox的位置向外padding一点确保目标尽可能被截出完整
    */
    
    int x_1,y_1,x_2,y_2,w_r,h_r,w_ori,h_ori,pad; 
    x_1=(int)obj.rect.x;
    y_1=(int)obj.rect.y;
    w_r=(int)obj.rect.width;
    h_r=(int)obj.rect.height;
    x_2=x_1+w_r;
    y_2=y_1+h_r;
    w_ori=im_ori.cols;
    h_ori=im_ori.rows;

    if(padding) //使得检测出来的ROI扩展一下
    {
        pad=min(w_r,h_r);
        x_1=max(0,x_1-pad/3);
        y_1=max(0,y_1-pad/3);
        x_2=min(w_ori-1,x_2+pad/3);
        y_2=min(h_ori-1,y_2+pad/3);
    
    }
    im_ori(cv::Rect(x_1, y_1,x_2-x_1, y_2-y_1)).copyTo(im_crop);
    int n_class=obj.label;
    if(n_class<4)
    {
    ID_type="obverse"; //正面
    }
    else
    {
    ID_type="reverse"; //反面
    }
    if(n_class==3 || n_class==7)
    {
        cv::transpose(im_crop,im_crop); //逆时针旋转90
        cv::flip(im_crop,im_crop,1);  //水平翻转
    }
    else if(n_class == 2 || n_class ==6)
    {
        cv::flip(im_crop,im_crop,-1);  //水平垂直翻转
    
    }
    else if(n_class == 1 || n_class ==5)
    {
    
        cv::transpose(im_crop,im_crop); //逆时针旋转90
        cv::flip(im_crop,im_crop,0);  //垂直翻转
    }   
    return 0;
}
int GetImgByEdge(cv::Mat& im_p,std::string& ID_type,const cv::Mat im_c,std::vector<Object>& edgeCoords,bool& perspective)
{
    /*
    im_p:
    ID_type:
    im_c: 原图像经过GetCropImageAndIDType扣除扩充，并角度校正后的图像
    edgeCoords: 边缘检测框的struct
    perspective: 标志是否经过仿射变换
    */
    perspective=false;
    typedef map<char, vector<vector<float>>> Eclass;
    Eclass eclass; 
    for(int i=0;i<edgeCoords.size();i++)
    {
        vector<float> xy_c;
        const Object& obj = edgeCoords[i];
        char key=obj.label+'0'; //int转换成char类型
        int x_1=(int)obj.rect.x;
        int y_1=(int)obj.rect.y;
        int w_r=(int)obj.rect.width;
        int h_r=(int)obj.rect.height;
        int x_2=x_1+w_r;
        int y_2=y_1+h_r;
        float x_c=(x_1+x_2)/2.0;
        float y_c=(y_1+y_2)/2.0;
        xy_c.push_back(x_c);
        xy_c.push_back(y_c);
        eclass[key].push_back(xy_c);  
    }
    //四个角点
    if((int)(eclass['0'].size())==1 && (int)(eclass['1'].size())==1 && (int)(eclass['2'].size())==1 && (int)(eclass['3'].size())==1)
    {
        vector<float>coord;
        for(int i=0;i<4;i++)
        {
            char c=i+'0';
            for (vector<vector<float>>::iterator itv =eclass[c].begin();itv != eclass[c].end();itv++)//通过迭代器访问数据
            {
                for (vector<float>::iterator vit = (*itv).begin();vit != (*itv).end();vit++)
                {
                    coord.push_back(*vit);
                }
            }
        
        } 
        cv::Mat coord_point=cv::Mat(coord,true); //8*1
        im_p=get_perspective_image(im_c,coord_point);
        perspective=true;
        return 0;
    }
    
    //两点确定一条直线
    if(((int)(eclass['0'].size())+(int)(eclass['1'].size())+(int)(eclass['4'].size()))>=2 && ((int)(eclass['1'].size())+(int)(eclass['2'].size())+
      (int)(eclass['6'].size()))>=2 && ((int)(eclass['2'].size())+(int)(eclass['3'].size())+(int)(eclass['5'].size()))>=2 && ((int)(eclass['3'].size())+
      (int)(eclass['0'].size())+(int)(eclass['7'].size()))>=2)
    {
        
    
        vector<vector<float>>line1,line2,line3,line4;
        
        if((int)(eclass['0'].size())==1)
        {
            line1.push_back(eclass['0'][0]);
        }
        if((int)(eclass['1'].size())==1)
        {
            line1.push_back(eclass['1'][0]);
        }
        
        for(int i=0;i<(int)(eclass['4'].size());i++)
        {
            line1.push_back(eclass['4'][i]);
        
        }
        
         if((int)(eclass['1'].size())==1)
        {
            line2.push_back(eclass['1'][0]);
        }
        if((int)(eclass['2'].size())==1)
        {
            line2.push_back(eclass['2'][0]);
        }
        
        for(int i=0;i<(int)(eclass['6'].size());i++)
        {
            line2.push_back(eclass['6'][i]);
        
        }
        
          if((int)(eclass['2'].size())==1)
        {
            line3.push_back(eclass['2'][0]);
        }
        if((int)(eclass['3'].size())==1)
        {
            line3.push_back(eclass['3'][0]);
        }
        
        for(int i=0;i<(int)(eclass['5'].size());i++)
        {
            line3.push_back(eclass['5'][i]);
        
        }
        
          if((int)(eclass['3'].size())==1)
        {
            line4.push_back(eclass['3'][0]);
        }
        if((int)(eclass['0'].size())==1)
        {
            line4.push_back(eclass['0'][0]);
        }
        
        for(int i=0;i<(int)(eclass['7'].size());i++)
        {
            line4.push_back(eclass['7'][i]);
        
        }
        if(int(line1.size())>1 && int(line2.size())>1 && int(line3.size())>1 &&int(line4.size())>1)
        {
            vector<float>angleC_4;
            vector<float>point;
            vector<vector<float>>angleCoords;
            bool target_r1=false,target_r2=false,target_r3=false,target_r4=false;   
            float *k1_b1=linear_regression(line1,target_r1); //线性会归根据坐标点拟合出直线
            float *k2_b2=linear_regression(line2,target_r2);
            float *k3_b3=linear_regression(line3,target_r3);
            float *k4_b4=linear_regression(line4,target_r4);

//             cout<<"k1:"<<k1_b1[1]<<"b1:"<<k1_b1[0]<<endl;
//             cout<<"k2:"<<k2_b2[1]<<"b2:"<<k2_b2[0]<<endl;
//             cout<<"k3:"<<k3_b3[1]<<"b3:"<<k3_b3[0]<<endl;
//             cout<<"k4:"<<k4_b4[1]<<"b4:"<<k4_b4[0]<<endl;
            bool point_is_exist=false;
            point= crosspoint(k1_b1,k4_b4,line1[0][0],line1[1][0],target_r1,target_r4,point_is_exist);
            if(point_is_exist)
            {
                angleCoords.push_back(point);
            }
            point= crosspoint(k1_b1,k2_b2,line2[0][0],line2[1][0],target_r1,target_r2,point_is_exist);
            if(point_is_exist)
            {
                angleCoords.push_back(point);
            }
            point= crosspoint(k2_b2,k3_b3,line3[0][0],line3[1][0],target_r2,target_r3,point_is_exist);
            if(point_is_exist)
            {
                angleCoords.push_back(point);
            }
            point= crosspoint(k3_b3,k4_b4,line4[0][0],line4[1][0],target_r3,target_r4,point_is_exist);
            if(point_is_exist)
            {
                angleCoords.push_back(point);
            }
           for(int i=0;i<(int)(angleCoords.size());i++)
           {
               for(int j=0;j<int(angleCoords[i].size());j++)
               {
                    angleC_4.push_back(angleCoords[i][j]);
               }
           }
           if((int)(angleC_4.size())==8)
            {
                cv::Mat angleC_4_mat=cv::Mat(angleC_4,true); //8*1
                cv::Mat im_an=get_perspective_image(im_c,angleC_4_mat);
//                 cv::imwrite("./im_angle.jpg",im_an);
                cv::Mat template_img;
                if(ID_type=="obverse")
                {
                   template_img=cv::imread("/src/notebooks/ImgIDRecogProj/utils/templateZ.jpg",1);
                }
               else
               {
                   template_img=cv::imread("/src/notebooks/ImgIDRecogProj/utils/templateF.jpg",1);
               }
               if(ID_type=="obverse")
               {
                   if(JudgeImgPSuccessZ(im_an,template_img))
                   {
                       perspective=true;
                   }
               }
               else
               {
                   if(JudgeImgPSuccessF(im_an,template_img))
                   {
                       perspective=true;
                   }
                   
               }
               return 0;
               
            }

        }
 
    }
    return 0; 
}
///SIFT变换
cv::Mat imresize(const cv::Mat &img, int img_max_size)
{
    int max_size = max(img.cols, img.rows);
    float ratio_re = (float) img_max_size / max_size;
    int w = (int) ((ratio_re * img.cols)/32) *  32;
    int h = (int) ((ratio_re * img.rows)/32) *  32;
    cv::Mat img2;
    cv::resize(img, img2, cv::Size(w, h));
//     cv::imwrite("/src/notebooks/ID_reg/hello_world/test_cpp.jpg",img2); 
    return img2;
}
cv::Mat SIFTInF(const cv::Mat &inpic_ori, const cv::Mat &template_img,float ratio,string img_type)
{
    cv::Mat inpic = imresize(inpic_ori,640);
    cv::Mat result;
    cv::Mat grayPic;
    cv::Mat graytemplate;
    cv::cvtColor(inpic, grayPic,cv::COLOR_BGR2GRAY);
    cv::cvtColor(template_img,graytemplate, cv::COLOR_BGR2GRAY);
    string obverse = "obverse";
    cv::Mat img1;
    cv::Mat img2;
    if (img_type==obverse){
        img1 = grayPic(cv::Range::all(),cv::Range(0,(int) grayPic.cols/2)).clone();
        img2 = graytemplate(cv::Range::all(),cv::Range(0,(int) graytemplate.cols*2/3)).clone();
    }
    else{
        img1 = grayPic.clone();
        img2 = template_img.clone();      
   } 
    auto sift = cv::SIFT::create();
    std::vector<cv::KeyPoint> keypoints1;
    std::vector<cv::KeyPoint> keypoints2; 
    cv::Mat descriptors1, descriptors2;
    
    sift->detectAndCompute(img1,cv::Mat(), keypoints1, descriptors1);
    sift->detectAndCompute(img2,cv::Mat(), keypoints2, descriptors2); 
//     特征匹配
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE);
//     KNN-NNDR匹配法
    std::vector<std::vector<cv::DMatch> > knn_matches;
    std::vector<cv::DMatch> good_matches;
    matcher->knnMatch( descriptors1, descriptors2, knn_matches, 2);
    for (auto & knn_matche : knn_matches) {
        if (knn_matche[0].distance < ratio * knn_matche[1].distance) {
            good_matches.push_back(knn_matche[0]);
        }
    }
    if (good_matches.size() > 4)
    {
        vector<cv::Point2f> src_pts;
        vector<cv::Point2f> dst_pts;
        for (size_t i=0;i<good_matches.size();i++)
        {
            src_pts.push_back(keypoints1[good_matches[i].queryIdx].pt);
            dst_pts.push_back(keypoints2[good_matches[i].trainIdx].pt);
        }
        cv::Mat H = cv::findHomography(src_pts,dst_pts,cv::RANSAC,10);
//         cout<<"H:"<<H<<endl;
//         cout<<"H:"<<H.cols<<endl;
        H.convertTo(H, CV_32F);
//         cout<<"H:"<<H.at<float>(0,0)<<endl;
//         cout<<"H:"<<H.at<float>(1,1)<<endl;
        
        if (H.cols > 0){
            if ((H.at<float>(0,0) > 0.5 && H.at<float>(0,0) < 2)&&(H.at<float>(1,1) > 0.5 && H.at<float>(1,1) < 2))
            {
                cv::warpPerspective(inpic, result, H, cv::Size(template_img.cols, template_img.rows));
                
            } 
        }
        
    }
    
    return result;
}
cv::Mat getPerImgInF(const cv::Mat &inpic,string img_type,bool &get_perspect)
{
    string obverse = "obverse";
    cv::Mat result;
    cv::Mat template_img;

    if (img_type==obverse){
        template_img = cv::imread("/src/notebooks/ImgIDRecogProj/utils/templateZ.jpg");
    }
    else{
        template_img = cv::imread("src/notebooks/ImgIDRecogProj/utils/templateF.jpg");
    }
    for( float ratio = 0.7; ratio < 1; ratio = ratio + 0.1)
    {
        try
        {
            result = SIFTInF(inpic, template_img, ratio,img_type);
        }
        catch(exception e){
            get_perspect = false;
            result = inpic.clone();
            return result;
        }
        
        if (result.cols>10)
        {
            if (img_type==obverse){
//                 cout<<JudgeImgPSuccessZ(template_img,result)<<endl;
                if (JudgeImgPSuccessZ(template_img,result)){
                    get_perspect = true;
                    return result;
                }
                else
                    continue;
            }
            else
            {
                if (JudgeImgPSuccessF(template_img,result))
                {
                    get_perspect = true;
                    return result;
                }
                else
                    continue;                
            }
            
        }
    }
    return inpic;
}
int fixbox(std::vector<vector<float>>& textbboxes,std::vector<int>& nclasses,std::vector<Object>& textbbox)
{
    for(int i=0;i<(int)(textbbox.size());i++)
    {
        vector<float> point_x1y1_x2y2;
        const Object& textobj = textbbox[i];
        float x_1=textobj.rect.x;
        float y_1=textobj.rect.y;
        float w_r=textobj.rect.width;
        float h_r=textobj.rect.height;
        float x_2=x_1+w_r;
        float y_2=y_1+h_r;
        point_x1y1_x2y2.push_back(x_1);
        point_x1y1_x2y2.push_back(y_1);
        point_x1y1_x2y2.push_back(x_2);
        point_x1y1_x2y2.push_back(y_1);
        point_x1y1_x2y2.push_back(x_2);
        point_x1y1_x2y2.push_back(y_2);
        point_x1y1_x2y2.push_back(x_1);
        point_x1y1_x2y2.push_back(y_2);
        
        textbboxes.push_back(point_x1y1_x2y2);
        nclasses.push_back(textobj.label);
    }
    return 0;
}
//text_detect后扣取图像部分三个相关函数,找出最贴近的外界矩阵
float cal_distance(float x1,float y1,float x2,float y2)
{
    return sqrt(pow((x2-x1),2)+pow((y2-y1),2));

}
std::pair<float,float> cal_width_height(cv::Mat coord)
{
    
    std::pair<float,float>wh;
    float width,height;
    float x1,y1,x2,y2,x3,y3;
    x1=coord.at<float>(0,0);
    y1=coord.at<float>(1,0);
    x2=coord.at<float>(2,0);
    y2=coord.at<float>(3,0);
    x3=coord.at<float>(4,0);
    y3=coord.at<float>(5,0);
    
    width=cal_distance(x1,y1,x2,y2);
    height=cal_distance(x2,y2,x3,y3);
//     cout<<"cal_width_height"<<endl;
//     cout<<width<<" "<<height<<endl;
    wh.first=width;
    wh.second=height;
    
    return wh;
}
cv::Mat get_perspective_image_rec_result(cv::Mat im_c,cv::Mat& coord)
{
    /*
        im_c: 经过text detect透视变换的图像
        coord:text detect的bbox
    
    */
    
    cv::Mat im_p;
    float width,height;
    std::pair<float,float>wh;
    wh=cal_width_height(coord);
    width=wh.first;
    height=wh.second;
//     cout<<width<<" "<<height<<endl;
    float a[4][2]={{0.0,0.0},{width,0.0},{width,height},{0.0,height}};
    cv::Mat pts1(4,2,CV_32FC1,a);
    cv::Mat pts2 = coord.reshape(2,4);
//     cout<<pts1<<endl;
//     cout<<pts2<<endl;
    cv::Mat H=cv::getPerspectiveTransform(pts2,pts1);
    cv::warpPerspective(im_c,im_p,H,cv::Size((int)(width),(int)(height)));
//     cv::imwrite("im_ppppppppp.jpg",im_p);
    return im_p;

}
void nclassindex(vector<int>& vec,vector<int>& res,int nclass)
{
    for(int i=0;i<vec.size();i++)
    {
        if(vec[i]==nclass)
        {
         res.push_back(i);
        }
    
    }
}
//正则化匹配
string matchString(string pattern_get,string content)
{
    
    regex pattern(pattern_get);
    smatch matchResult;  //正则化匹配string类型
    string matchString="";
    //迭代器声明
    string::const_iterator iterStart = content.begin();
    string::const_iterator iterEnd = content.end();

    while (regex_search(iterStart, iterEnd, matchResult, pattern))
    {
        matchString += matchResult[0];
        iterStart = matchResult[0].second;	//更新搜索起始位置,搜索剩下的字符串
    } 
    return matchString;

}

// # 单字民族对应的相似字符, 用于辅助纠正
void get_similar_sex(map<string,vector<string>>& similar_sex){
    vector<string>similar_sex1={"墨","奥","速","隽","贵","美","期","勇"};
    similar_sex.insert(pair<string,vector<string>>("男",similar_sex1));//["男"].push_back(similar_sex1);
    vector<string>similar_sex2={"因","安","汝","文","委"};
    similar_sex.insert(pair<string,vector<string>>("女",similar_sex2));//["女"].push_back(similar_sex2);
//     cout<<"sex: "<<similar_sex["男"].size()<<" ";    
}

void get_similar(map<string,vector<string>>& similar_data_nation,vector<string>&nationality){
    
    nationality={"汉","彝","侗","蒙古","回","藏","维吾尔",
                  "苗","壮","朝鲜","满","瑶","白","土家",
                  "哈尼","哈萨克","黎","傈僳","佤","畲","高山",
                  "拉祜","水","东乡","纳西","景颇","柯尔克孜",
                  "土","达斡尔","羌","撒拉","毛难","仫佬",
                  "仡佬","锡伯","阿昌","普米","塔吉克","怒","乌孜别克",
                  "俄罗斯","德昂","保安","裕固","崩龙","独龙","鄂伦春","赫哲",
                  "门巴","珞巴","基诺","鄂温克","傣","京","塔塔尔","布朗","布依"
                  };   
    
    vector<string>similar1={"固", "同", "倜", "调", "垌", "桐", "恫", "洞", "峒", "硐", "胴"};
    similar_data_nation.insert(pair<string,vector<string>>("侗",similar1));//["侗"].push_back(similar1);
    vector<string>similar2={"瞒", "蹒", "螨", "潢", "滿"};
    similar_data_nation.insert(pair<string,vector<string>>("满",similar2));//["满"].push_back(similar2);
    vector<string>similar3={"汶", "汊", "汝", "汐", "汲", "汀", "波", "叹", "仅", "汊"};
    similar_data_nation.insert(pair<string,vector<string>>("汉",similar3));//["汉"].push_back(similar3);
    vector<string>similar4={"惊", "凉", "谅", "掠", "谅", "晾", "掠", "景", "亰"};
    similar_data_nation.insert(pair<string,vector<string>>("京",similar4));//["京"].push_back(similar4);
    vector<string>similar5={"泰", "秦", "倴", "僚", "溙", "奉"};
    similar_data_nation.insert(pair<string,vector<string>>("傣",similar5));//["傣"].push_back(similar5);

    vector<string>similar6={"亩", "启", "茔", "芸", "盅", "电", "宙", "田", "喵", "描", "猫"};
    similar_data_nation.insert(pair<string,vector<string>>("苗",similar6));//["苗"].push_back(similar6);
    vector<string>similar7={"臧", "臧", "葬"};
    similar_data_nation.insert(pair<string,vector<string>>("藏",similar7));//["壮"].push_back(similar7);
    vector<string>similar8={"壯", "莊", "荘", "妆"};
    similar_data_nation.insert(pair<string,vector<string>>("壮",similar8));//["壮"].push_back(similar7);


    vector<string>similar9={"摇", "遥", "谣"};
    similar_data_nation.insert(pair<string,vector<string>>("瑶",similar9));//["瑶"].push_back(similar8);
    vector<string>similar10={"日", "自", "百", "囱", "曰", "囪", "甶", "凹", "汩", "彐",
              "旧", "囗", "田", "帕", "伯", "拍", "泊", "柏", "陌"};
    similar_data_nation.insert(pair<string,vector<string>>("白",similar10));//["白"].push_back(similar9);

    vector<string>similar11={"藜", "棃", "黧", "梨", "犁", "藜"};
    similar_data_nation.insert(pair<string,vector<string>>("黎",similar11));//["黎"].push_back(similar10);
    vector<string>similar12={"仉", "巩", "讥", "伉", "瓦", "咓", "砙"};
    similar_data_nation.insert(pair<string,vector<string>>("佤",similar12));//["佤"].push_back(similar11);

    vector<string>similar13={"番", "禽", "肏"};
    similar_data_nation.insert(pair<string,vector<string>>("畲",similar13));//["畲"].push_back(similar12);
    vector<string>similar14={"氷", "囦", "永", "冰", "木", "未"};
    similar_data_nation.insert(pair<string,vector<string>>("水",similar14));//["水"].push_back(similar13);

    vector<string>similar15={"工", "二", "三", "王", "亍", "士", "七"};
    similar_data_nation.insert(pair<string,vector<string>>("土",similar15));//["土"].push_back(similar14);
    vector<string>similar16={"恙", "羊", "羔", "羔"};
    similar_data_nation.insert(pair<string,vector<string>>("羌",similar16));//["羌"].push_back(similar15);

    vector<string>similar17={"囚", "四", "迴", "佪", "廻", "洄", "叵", "固", "国","间", "囧", "区","而"};
    similar_data_nation.insert(pair<string,vector<string>>("回",similar17));//["回"].push_back(similar17);
    vector<string>similar18={"努", "恕", "奴", "弩", "驽", "孥", "㐐"};
    similar_data_nation.insert(pair<string,vector<string>>("怒",similar18));//["怒"].push_back(similar18);   
}
template <typename T>
std::vector<size_t> sort_indexes(const std::vector<T> &v)
{
  // 初始化索引向量
  std::vector<size_t> idx(v.size());
  //使用iota对向量赋0~？的连续值
  std::iota(idx.begin(), idx.end(), 0);
  // 通过比较v的值对索引idx进行排序
  std::sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) { return v[i1] < v[i2]; });
  return idx;
}

vector<string> split_utf8(string str){
    string strChar;
    vector<string> str_s;
    for(int i = 0; str[i] != '\0'; )
    {
        char chr = str[i];
//         cout << chr <<endl;
        if((chr & 0x80) == 0)
        {
            strChar = str.substr(i,1);
            ++i;
        }//chr是1111 1xxx
        else if((chr & 0xF8) == 0xF8)
        {
            strChar = str.substr(i, 5);
            i+=5;
        }//chr是1111 xxxx
        else if((chr & 0xF0) == 0xF0)
        {
            strChar = str.substr(i, 4);
            i+=4;
        }//chr是111x xxxx
        else if((chr & 0xE0) == 0xE0)
        {
            strChar = str.substr(i, 3);
            i+=3;
        }//chr是11xx xxxx
        else if((chr & 0xC0) == 0xC0)
        {
            strChar = str.substr(i, 2);
            i+=2;
        }
        str_s.push_back(strChar);
    }
    return str_s;
}
string vecTostring(const vector<string> vec,size_t start,size_t end)
{
    string res="";
    for (size_t i=start;i<min(end,vec.size());i++){
        res += vec[i];
    }
    return res;   
}

bool search(const vector<string> strs, string target)
{
//     vector<string> ::iterator t;
    auto t = find(strs.begin(),strs.end(),target);
    if(t != strs.end()){
        return true;
    }
    return false;
}

string judge_nationality(string data_n){
    string result = "";
    map<string,vector<string>> similar_data_nation;
    vector<string>nationality;
    get_similar(similar_data_nation,nationality);  
//     cout<<"nationality: "<<nationality[0]<<" ";
    vector<float> scores;
    
    if (count(nationality.begin(),nationality.end(),data_n)){
        return data_n;
    }
    else{
        vector<string>data_n_s = split_utf8(data_n);
        if (data_n_s.size() == 1)
        {
            for(map<string,vector<string>>::iterator iter = similar_data_nation.begin(); iter != similar_data_nation.end(); ++iter)
            {
                string nation_key = iter->first;
//                 cout<<"key:"<<nation_key;
                vector<string>similar_nation = iter->second;
                for (int i=0;i<similar_nation.size();i++)
                {
//                     cout<<similar_nation[i]<<" "; 
                    if (similar_nation[i]==data_n)
                    {
//                         cout<<"modify:"<<nation_key<<" ";
                        return nation_key;
                    }
                }
//                 cout<<endl;
            }
        }
        for (int num=0;num<nationality.size();num++)
        {
            vector<string>nation = split_utf8(nationality[num]);
            float score = 0;
            for (int idx_d=0;idx_d<data_n_s.size();idx_d++)
            {
                for(int idx_n=0;idx_n<nation.size();idx_n++)
                {
                    if (nation[idx_n] == data_n_s[idx_d])
                    {
                        if (idx_d == idx_n)
                        {
                            score += 100;
                        }
                        else
                        {
                            score += 50;
                        }
                        break;
                    }
                }  
            }
            if (data_n_s.size() == nation.size())
            {
                score += 70;
            }
            scores.push_back(score);
        }
    }
    auto index_ = sort_indexes(scores);
    return nationality[index_[index_.size()-1]];
}
string fix_sex(string sex_str)
{
    if (sex_str=="男" || sex_str=="女")
    {
        return sex_str;
    }
    else{
        map<string,vector<string>> similar_sex;
        get_similar_sex(similar_sex);
        for(map<string,vector<string>>::iterator iter = similar_sex.begin(); iter != similar_sex.end(); ++iter)
        {
            string sex_key = iter->first;
//             cout<<"key:"<<sex_key;
            vector<string>similar_sex = iter->second;
            for (int i=0;i<similar_sex.size();i++)
            {
//                 cout<<similar_sex[i]<<" "; 
                if (similar_sex[i]==sex_str)
                {
//                     cout<<"modify sex:"<<sex_key<<" ";
                    return sex_key;
                }
            }
        }        
      }
    
}
void get_result_observe(ID_Result_Obverse *Obverse_result,vector<int>&nclasses,vector<vector<float> >& bbox,vector<string>&reg_result)
{
    /*
        Obverse_result：返回后处理校正的识别结果
        nclasses：所有text bbox对应的label
        bbox：所有text的bbox的值
        reg_result:所有识别结果
    */
    if (reg_result.size()>0){
        //name
        vector<int>index_class0;
        nclassindex(nclasses,index_class0,0);
        if (index_class0.size()==1){
            string class0_pattern = "[a-zA-Z0-9\u4e00-\u9fa5·]+";
            Obverse_result->name = matchString(class0_pattern,reg_result[index_class0[0]]);
            Obverse_result->name_coord = bbox[index_class0[0]];
        }
        
        //year
        vector<int>index_class1;
        nclassindex(nclasses,index_class1,1);     
        if (index_class1.size()==1){
            string class1_pattern = "[0-9]+";
            Obverse_result->year = matchString(class1_pattern,reg_result[index_class1[0]]); 
            Obverse_result->year_coord = bbox[index_class1[0]];
        }            

        //month
        vector<int>index_class2;
        nclassindex(nclasses,index_class2,2);     
        if (index_class2.size()==1){
            string class2_pattern = "[0-9]+";
            Obverse_result->month = matchString(class2_pattern,reg_result[index_class2[0]]); 
            Obverse_result->month_coord = bbox[index_class2[0]];
        } 
        
        //day
        vector<int>index_class3;
        nclassindex(nclasses,index_class3,3);     
        if (index_class3.size()==1){
            string class3_pattern = "[0-9]+";
            Obverse_result->day = matchString(class3_pattern,reg_result[index_class3[0]]); 
            Obverse_result->day_coord = bbox[index_class3[0]];
        }
        
        //birth
        Obverse_result->birth = Obverse_result->year+"."+Obverse_result->month+"."+Obverse_result->day;
        vector<vector<float> > birth_all(8,vector<float>(3,0));
        
        for(int i=0;i<birth_all.size();i++){
           birth_all[i][0] = Obverse_result->year_coord[i];
           birth_all[i][1] = Obverse_result->month_coord[i];
           birth_all[i][2] = Obverse_result->day_coord[i];
        }
        Obverse_result->birth_coord={
           *min_element(birth_all[0].begin(),birth_all[0].end()),
           *min_element(birth_all[1].begin(),birth_all[1].end()),
           *max_element(birth_all[2].begin(),birth_all[2].end()),
           *min_element(birth_all[3].begin(),birth_all[3].end()),
           *max_element(birth_all[4].begin(),birth_all[4].end()),
           *max_element(birth_all[5].begin(),birth_all[5].end()),
           *min_element(birth_all[6].begin(),birth_all[6].end()),
           *max_element(birth_all[7].begin(),birth_all[7].end())
           };
        
        //national                
        vector<int>index_class4;
        nclassindex(nclasses,index_class4,4);     
        if (index_class4.size()==1)
        {
            string class4_pattern = "[\u4e00-\u9fa5]+";
            string national_match = matchString(class4_pattern,reg_result[index_class4[0]]); 
            if (national_match=="穿青人"){
                Obverse_result->national = national_match; 
                Obverse_result->national_coord = bbox[index_class4[0]];               
            }
            else{
                Obverse_result->national = judge_nationality(national_match); 
                Obverse_result->national_coord = bbox[index_class4[0]];                 
            }
            
        } 
    
        //sex
        vector<int>index_class5;
        nclassindex(nclasses,index_class5,5);     
        if (index_class5.size()==1){
            string class5_pattern = "[\u4e00-\u9fa5]+";
            string sex_match = matchString(class5_pattern,reg_result[index_class5[0]]); 
            Obverse_result->sex = fix_sex(sex_match);
            Obverse_result->sex_coord = bbox[index_class5[0]];
        }  
        
        //address
        vector<int>index_class6;
        nclassindex(nclasses,index_class6,6);     
        if (index_class6.size()>=1)
        {
            vector<vector<float> > address_all(8,vector<float>(index_class6.size(),0));

            for(int i=0;i<address_all.size();i++)
            {
               for (int box_index=0;box_index<index_class6.size();box_index++)
               {
                   address_all[i][box_index] = bbox[index_class6[box_index]][i];
               }
            }
            auto index_ = sort_indexes(address_all[1]);
            string content = "";

            for(int i=0;i<index_.size();i++){
                content += reg_result[index_class6[index_[i]]];
            }
            string class6_pattern = "[—a-zA-Z0-9\u4e00-\u9fa5-.·]+"; //—
            Obverse_result->address = matchString(class6_pattern,content);
            Obverse_result->address_coord = {
                *min_element(address_all[0].begin(),address_all[0].end()),
                *min_element(address_all[1].begin(),address_all[1].end()),
                *max_element(address_all[2].begin(),address_all[2].end()),
                *min_element(address_all[3].begin(),address_all[3].end()),
                *max_element(address_all[4].begin(),address_all[4].end()),
                *max_element(address_all[5].begin(),address_all[5].end()),
                *min_element(address_all[6].begin(),address_all[6].end()),
                *max_element(address_all[7].begin(),address_all[7].end())
                };
        }                      
                    
        //id_number
        vector<int>index_class7;
        nclassindex(nclasses,index_class7,7);     
        if (index_class7.size()==1){
            string class7_pattern = "[0-9Xx]+";
            string id_number = matchString(class7_pattern,reg_result[index_class7[0]]);
            Obverse_result->id_number = id_number;
            Obverse_result->id_number_coord = bbox[index_class7[0]];
            //根据身份证号码的倒数第二位判断性别
            if (id_number.length()==18)
            {
                if (id_number[id_number.length()-2] %2 == 0)
                {
                   Obverse_result->sex = "女"; 
                }
                else
                {
                  Obverse_result->sex = "男";  
                }
            }
        } 
    }
    
}
void get_result_reserve(ID_Result_Reverse *Reverse_result,vector<int>&nclasses,vector<vector<float> >& bbox,vector<string>&reg_result)
{
    if (reg_result.size()>0)
    {
        //issued
        vector<int>index_class8;
        nclassindex(nclasses,index_class8,8);//取出标签为8的识别结果，是要是issued
//         cout<<"issued检测："<<index_class8.size()<<endl;
        if (index_class8.size()==1)
        {
            string class8_pattern = "[\u4e00-\u9fa5]+";//汉字
            string issued_match = matchString(class8_pattern,reg_result[index_class8[0]]);
            Reverse_result->issued = issued_match;
            vector<string>issued_match_s = split_utf8(issued_match);
            if (issued_match_s.size() > 2)
            {
                if (issued_match_s[issued_match_s.size()-2]=="分")
                {
                    issued_match_s[issued_match_s.size()-2] = "分";
                    issued_match_s[issued_match_s.size()-1] = "局";
                    Reverse_result->issued = vecTostring(issued_match_s,0,issued_match_s.size());
                }
            }
            Reverse_result->issued_coord = bbox[index_class8[0]];
        }   
        //effective_date
        vector<int>index_class9;
        nclassindex(nclasses,index_class9,9);  
//         cout<<index_class9.size()<<endl;
        if (index_class9.size()==1)
        {
            string class9_pattern = "[0-9长期]+";
            string effective_date_match = matchString(class9_pattern,reg_result[index_class9[0]]); 
            vector<string>effective_date_match_s = split_utf8(effective_date_match);
            if (effective_date_match_s.size()==16)
            {
                Reverse_result->effective_date = 
                    vecTostring(effective_date_match_s,0,4)+"."
                    +vecTostring(effective_date_match_s,4,6)+"."
                    +vecTostring(effective_date_match_s,6,8)+"-"
                    +vecTostring(effective_date_match_s,8,12)+"."
                    +vecTostring(effective_date_match_s,12,14)+"."
                    +vecTostring(effective_date_match_s,14,effective_date_match_s.size());
            }
            else if (search(effective_date_match_s, "长") || search(effective_date_match_s, "期"))
            {
                effective_date_match = matchString("[0-9]+",effective_date_match);
                vector<string>effective_date_match_s = split_utf8(effective_date_match);
                if (effective_date_match_s.size()==8)
                {
                    Reverse_result->effective_date = 
                        vecTostring(effective_date_match_s,0,4)+"."
                        +vecTostring(effective_date_match_s,4,6)+"."
                        +vecTostring(effective_date_match_s,6,8)+"-"+"长期";                    
                }
            }
            else
            {
                Reverse_result->effective_date = effective_date_match;
            }
            Reverse_result->effective_date_coord = bbox[index_class9[0]];
        }  
//         cout<<"lalalalalalal"<<endl;
    }      
}