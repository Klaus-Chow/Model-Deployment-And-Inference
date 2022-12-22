
#include <iostream>
#include <map>
#include <string>
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp> 
#include <opencv2/features2d/features2d.hpp>
#include <vector>
#include <numeric>

// #include "siftdispatch.cpp"
using namespace std;



template <class T>
void display(string str, cv::Mat& src) //将矩阵数据进行展示
{
	cout << str << endl;
	int cols = src.cols;
	int rows = src.rows;
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			cout << src.at<T>(i, j) << ",";
		}
		cout << endl;
	}
}


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
        std::cout << "cut_template sum： " << cv::sum(cut_template) << std::endl;
        std::cout << "cut_template mean： " << cv::mean(cut_template) << std::endl; 

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
bool JudgeImgPSuccessZ(const cv::Mat &template_img, const cv::Mat &result, int numthresh=5, int meanthresh=15){

    
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
        std::cout << "cut_template sum： " << cv::sum(cut_template) << std::endl;
        std::cout << "cut_template mean： " << cv::mean(cut_template) << std::endl; 
        if (get_mean[0] < meanthresh){
            flag += 1;
            } 
    }
    if (flag >= numthresh){
        return true;
        }
    return false;      
}

cv::Mat imresize(const cv::Mat &img, int img_max_size){
    int max_size = max(img.cols, img.rows);
    float ratio_re = (float) img_max_size / max_size;
    int w = (int) ((ratio_re * img.cols)/32) *  32;
    int h = (int) ((ratio_re * img.rows)/32) *  32;
    cv::Mat img2;
    cv::resize(img, img2, cv::Size(w, h));
//     cv::imwrite("/src/notebooks/ID_reg/hello_world/test_cpp.jpg",img2); 
    return img2;
}

cv::Mat SIFTInF(const cv::Mat &result, const cv::Mat &template_img,float ratio,string img_type){
    cv::Mat inpic = imresize(result,640);
    
    cv::Mat grayPic;
    cv::Mat graytemplate;
    cv::cvtColor(inpic, grayPic,cv::COLOR_BGR2GRAY);
    cv::cvtColor(template_img,graytemplate, cv::COLOR_BGR2GRAY);
    string obverse = "obverse";
    if (img_type==obverse){
        cv::Mat img1 = grayPic(cv::Range::all(),cv::Range(0,(int) grayPic.cols/2)).clone();
        cv::Mat img2 = graytemplate(cv::Range::all(),cv::Range(0,(int) graytemplate.cols*2/3)).clone();
    }
    else{
        cv::Mat img1 = grayPic.clone();
        cv::Mat img2 = template_img.clone();      
        }
//     计算特征点
//     cv::SIFT sift;
//     auto sift = cv::SIFT::create();
    std::vector<cv::KeyPoint> keypoints1;
    std::vector<cv::KeyPoint> keypoints2;    
//     sift->detect(img1, keypoints1);
//     sift->detect(img2, keypoints2);
// //     计算特征描述符
//     cv::Mat descriptors1, descriptors2;
//     sift->compute(img1, keypoints1, descriptors1);
//     sift->compute(img2, keypoints2, descriptors2);
// //     特征匹配
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE);
// //     KNN-NNDR匹配法
    std::vector<std::vector<cv::DMatch> > knn_matches;
    std::vector<cv::DMatch> good_matches;
//     matcher->knnMatch( descriptors1, descriptors2, knn_matches, 2);
//     for (auto & knn_matche : knn_matches) {
//         if (knn_matche[0].distance < ratio * knn_matche[1].distance) {
//             good_matches.push_back(knn_matche[0]);
//         }
//     }
    return inpic;
 
}

cv::Mat getPerImgInF(const cv::Mat &inpic,string img_type,bool &get_perspect)
{
    string obverse = "obverse";
    cv::Mat result;
    cv::Mat template_img;

    if (img_type==obverse){
        template_img = cv::imread("/src/notebooks/ID_reg/hello_world/test_img/templateZ.jpg");
        std::cout << "宽度： "<< template_img.cols << std::endl;
    }
    else{
        template_img = cv::imread("/src/notebooks/ID_reg/hello_world/test_img/templateF.jpg");
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

float* linear_regression(vector<vector<float> >& line,bool &get_kb)
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
        cout << "x: " << x2[i] << " ";
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
    
// int main(int argc, char** argv){
//     int N=3, M=2; 
//     vector<vector<float> > obj(N, vector<float>(M)); //定义二维动态数组5行6列  
//     obj[0][0] = 143;
//     obj[0][1] = 479;
//     obj[1][0] = 315;
//     obj[1][1] = 306.5;
//     obj[2][0] = 501;
//     obj[2][1] = 122;

//     for(int i=0;i<N;i++)//方法一 
//     {
//         for (int j=0;j<obj[i].size();j++){
//             cout<<obj[i][j]<<" ";
//         }
//         cout<<"\n";
//     } 
//     bool get_kb=false;
//     float *result = linear_regression(obj,get_kb);
    
//     cout<<"get_kb: "<<get_kb<<" ";
//     cout<<"k: "<<result[0]<<" ";
//     cout<<"b: "<<result[1]<<" ";
//     cout<<"\n";
//     delete[] result;
// }


// //  JudgeImgPSuccessF JudgeImgPSuccessZ
int main(int argc, char** argv)
{
    string image_path,templateF_path;
    image_path = "/src/notebooks/ID_reg/hello_world/test_img/F.jpg";
    templateF_path = "/src/notebooks/ID_reg/hello_world/test_img/templateF.jpg";
    cv::Mat src_img = cv::imread(image_path);
    cv::Mat template_img = cv::imread(templateF_path);
//     src_img = cv::imread(image_path,1);
    //宽度
    std::cout << "宽度： "<< src_img.cols << std::endl;
    //高度
    std::cout << "高度： " << src_img.rows << std::endl;
    //通道数
    std::cout << "通道数： " << src_img.channels() << std::endl;
//     cv::Mat des_img;
//     cv::flip(src_img,des_img,0); //水平旋转180度
     //宽度
    std::cout << "宽度： "<< template_img.cols << std::endl;
    //高度
    std::cout << "高度： " << template_img.rows << std::endl;
    //通道数
    std::cout << "通道数： " << template_img.channels() << std::endl;
//     cv::imwrite("/src/notebooks/ID_reg/hello_world/test.jpg",des_img);   
    
    bool per= false;
    per = JudgeImgPSuccessF(src_img,template_img);
    std::cout << "per： " << per << std::endl;
//     cv::Mat src1 = (cv::Mat_<float>(2, 3) << 0, 255, 0, 255, 0, 0);
//     cv::Mat src2 = (cv::Mat_<float>(2, 3) << 0, 255, 255, 0, 0, 255);  
//     src1 = src1 - src2;
//     src1.convertTo(src1, CV_32F);
//     src1.setTo(1,src1 < 0);
//     display<float>("运算符矩阵相减", src1);
//     std::cout << "sum： " << cv::sum(src1) << std::endl;
//     std::cout << "mean： " << cv::mean(src1) << std::endl;
    return 0;
}
    
