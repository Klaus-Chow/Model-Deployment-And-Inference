
#include <iostream>
#include <map>
#include <string>
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp> 
#include <opencv2/features2d/features2d.hpp>
#include "opencv2/calib3d/calib3d.hpp" 
#include <vector>
#include <numeric>
using namespace std;
using namespace cv;


// template <class T>
// void display(string str, cv::Mat& src) //将矩阵数据进行展示
// {
// 	cout << str << endl;
// 	int cols = src.cols;
// 	int rows = src.rows;
// 	for (int i = 0; i < rows; i++)
// 	{
// 		for (int j = 0; j < cols; j++)
// 		{
// 			cout << src.at<T>(i, j) << ",";
// 		}
// 		cout << endl;
// 	}
// }


// // 有效期限 template_arr[0];签发机关[1];居民身份证[2];中华人民共和国[3]
// bool JudgeImgPSuccessF(const cv::Mat &template_img, const cv::Mat &result){
    
//     int template_arr[4][4] = {{197, 456, 313, 486},{197, 389, 313, 419},
//                                      {255, 149, 779, 224},{283, 49, 745, 112}
//                                     };
//     for (int i = 0; i < 4; i++){
//         template_arr[i][0] = (int) template_arr[i][0]/1.3375;
//         template_arr[i][1] = (int) template_arr[i][1]/1.2981;
//         template_arr[i][2] = (int) template_arr[i][2]/1.3375;
//         template_arr[i][3] = (int) template_arr[i][3]/1.2981;      
//     }

//     int flag = 0;
//     for (int j = 0; j < 4; j++){ 
//         cv::Mat cut_template,cut_result;
//         cut_template = template_img(cv::Range(template_arr[j][1],template_arr[j][3]),cv::Range(template_arr[j][0],template_arr[j][2])).clone(); 
//         cut_result = result(cv::Range(template_arr[j][1],template_arr[j][3]),cv::Range(template_arr[j][0],template_arr[j][2])).clone(); 
//         cv::cvtColor(cut_template, cut_template,cv::COLOR_RGB2GRAY);
//         cv::cvtColor(cut_result, cut_result,cv::COLOR_RGB2GRAY);
//         cv::threshold(cut_template,cut_template, 0, 255, cv::THRESH_OTSU);
//         cv::threshold(cut_result,cut_result, 0, 255, cv::THRESH_OTSU);
//         cut_template.convertTo(cut_template, CV_32F);
//         cut_result.convertTo(cut_result, CV_32F);
//         cut_template = cut_template - cut_result;
//         cut_template.setTo(1,cut_template < 0);
//         cut_template.convertTo(cut_template, CV_32F);
//         cv::Scalar get_mean;
//         get_mean = cv::mean(cut_template);
// //         std::cout << "cut_template sum： " << cv::sum(cut_template) << std::endl;
// //         std::cout << "cut_template mean： " << cv::mean(cut_template) << std::endl; 

//         if (get_mean[0] < 15){
//             flag += 1;
//             } 
//     }
//     if (flag >= 2){
//         return true;
//         }
//     return false;      
// }
    
// // 年 template_arr[0];月[1];日[2];姓名[3];性别[3];民族[3];出生[3];住址[3];身份证号码[3]
// bool JudgeImgPSuccessZ(const cv::Mat &template_img, const cv::Mat &result, int numthresh=5, int meanthresh=15){

    
//     int template_arr[9][4] = {{260,208,291,239},{337,208,368,239},
//                               {416,208,447,239},{70,81,143,112},
//                               {70,142,143,173},{259,144,332,175},
//                               {68,208,141,239},{71,278,142,309},
//                               {70,449,261,481}
//                                     };
//     for (int i = 0; i < 9; i++){
//         template_arr[i][0] = (int) template_arr[i][0]/1.3375;
//         template_arr[i][1] = (int) template_arr[i][1]/1.2981;
//         template_arr[i][2] = (int) template_arr[i][2]/1.3375;
//         template_arr[i][3] = (int) template_arr[i][3]/1.2981;      
//     }

//     int flag = 0;
//     for (int j = 0; j < 9; j++){ 
//         cv::Mat cut_template,cut_result;
//         cut_template = template_img(cv::Range(template_arr[j][1],template_arr[j][3]),cv::Range(template_arr[j][0],template_arr[j][2])).clone(); 
//         cut_result = result(cv::Range(template_arr[j][1],template_arr[j][3]),cv::Range(template_arr[j][0],template_arr[j][2])).clone(); 
//         cv::cvtColor(cut_template, cut_template,cv::COLOR_RGB2GRAY);
//         cv::cvtColor(cut_result, cut_result,cv::COLOR_RGB2GRAY);
//         cv::threshold(cut_template,cut_template, 0, 255, cv::THRESH_OTSU);
//         cv::threshold(cut_result,cut_result, 0, 255, cv::THRESH_OTSU);
//         cut_template.convertTo(cut_template, CV_32F);
//         cut_result.convertTo(cut_result, CV_32F);
//         cut_template = cut_template - cut_result;
//         cut_template.setTo(1,cut_template < 0);
//         cv::Scalar get_mean;
//         get_mean = cv::mean(cut_template);
// //         std::cout << "cut_template sum： " << cv::sum(cut_template) << std::endl;
// //         std::cout << "cut_template mean： " << cv::mean(cut_template) << std::endl; 
//         if (get_mean[0] < meanthresh){
//             flag += 1;
//             } 
//     }
//     if (flag >= numthresh){
//         return true;
//         }
//     return false;      
// }

// cv::Mat imresize(const cv::Mat &img, int img_max_size){
//     int max_size = max(img.cols, img.rows);
//     float ratio_re = (float) img_max_size / max_size;
//     int w = (int) ((ratio_re * img.cols)/32) *  32;
//     int h = (int) ((ratio_re * img.rows)/32) *  32;
//     cv::Mat img2;
//     cv::resize(img, img2, cv::Size(w, h));
// //     cv::imwrite("/src/notebooks/ID_reg/hello_world/test_cpp.jpg",img2); 
//     return img2;
// }

// cv::Mat SIFTInF(const cv::Mat &inpic_ori, const cv::Mat &template_img,float ratio,string img_type)
// {
//     cv::Mat inpic = imresize(inpic_ori,640);
//     cv::Mat result;
//     cv::Mat grayPic;
//     cv::Mat graytemplate;
//     cv::cvtColor(inpic, grayPic,cv::COLOR_BGR2GRAY);
//     cv::cvtColor(template_img,graytemplate, cv::COLOR_BGR2GRAY);
//     string obverse = "obverse";
//     cv::Mat img1;
//     cv::Mat img2;
//     if (img_type==obverse){
//         img1 = grayPic(cv::Range::all(),cv::Range(0,(int) grayPic.cols/2)).clone();
//         img2 = graytemplate(cv::Range::all(),cv::Range(0,(int) graytemplate.cols*2/3)).clone();
//     }
//     else{
//         img1 = grayPic.clone();
//         img2 = template_img.clone();      
//    } 
//     auto sift = cv::SIFT::create();
//     std::vector<cv::KeyPoint> keypoints1;
//     std::vector<cv::KeyPoint> keypoints2; 
//     cv::Mat descriptors1, descriptors2;
    
//     sift->detectAndCompute(img1,cv::Mat(), keypoints1, descriptors1);
//     sift->detectAndCompute(img2,cv::Mat(), keypoints2, descriptors2); 
// //     特征匹配
//     cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE);
// //     KNN-NNDR匹配法
//     std::vector<std::vector<cv::DMatch> > knn_matches;
//     std::vector<cv::DMatch> good_matches;
//     matcher->knnMatch( descriptors1, descriptors2, knn_matches, 2);
//     for (auto & knn_matche : knn_matches) {
//         if (knn_matche[0].distance < ratio * knn_matche[1].distance) {
//             good_matches.push_back(knn_matche[0]);
//         }
//     }
//     if (good_matches.size() > 4)
//     {
//         vector<cv::Point2f> src_pts;
//         vector<cv::Point2f> dst_pts;
//         for (size_t i=0;i<good_matches.size();i++)
//         {
//             src_pts.push_back(keypoints1[good_matches[i].queryIdx].pt);
//             dst_pts.push_back(keypoints2[good_matches[i].trainIdx].pt);
//         }
//         cv::Mat H = cv::findHomography(src_pts,dst_pts,cv::RANSAC,10);
// //         cout<<"H:"<<H<<endl;
// //         cout<<"H:"<<H.cols<<endl;
//         H.convertTo(H, CV_32F);
// //         cout<<"H:"<<H.at<float>(0,0)<<endl;
// //         cout<<"H:"<<H.at<float>(1,1)<<endl;
        
//         if (H.cols > 0){
//             if ((H.at<float>(0,0) > 0.5 && H.at<float>(0,0) < 2)&&(H.at<float>(1,1) > 0.5 && H.at<float>(1,1) < 2))
//             {
//                 cv::warpPerspective(inpic, result, H, cv::Size(template_img.cols, template_img.rows));
                
//             } 
//         }
        
//     }
    
//     return result;
// }

// cv::Mat getPerImgInF(const cv::Mat &inpic,string img_type,bool &get_perspect)
// {
//     string obverse = "obverse";
//     cv::Mat result;
//     cv::Mat template_img;

//     if (img_type==obverse){
//         template_img = cv::imread("/src/notebooks/ID_reg/hello_world/test_img/templateZ.jpg");
//     }
//     else{
//         template_img = cv::imread("/src/notebooks/ID_reg/hello_world/test_img/templateF.jpg");
//     }
//     for( float ratio = 0.7; ratio < 1; ratio = ratio + 0.1)
//     {
//         try
//         {
//             result = SIFTInF(inpic, template_img, ratio,img_type);
//         }
//         catch(exception e){
//             get_perspect = false;
//             result = inpic.clone();
//             return result;
//         }
        
//         if (result.cols>10)
//         {
//             if (img_type==obverse){
//                 cout<<JudgeImgPSuccessZ(template_img,result)<<endl;
//                 if (JudgeImgPSuccessZ(template_img,result)){
//                     get_perspect = true;
//                     return result;
//                 }
//                 else
//                     continue;
//             }
//             else
//             {
//                 if (JudgeImgPSuccessF(template_img,result))
//                 {
//                     get_perspect = true;
//                     return result;
//                 }
//                 else
//                     continue;                
//             }
            
//         }
//     }
//     return inpic;
// }

int main(int argc, char** argv)
{
//     string image_path,templateF_path;
//     image_path = "/src/notebooks/ID_reg/hello_world/test_img/F.jpg";
//     templateF_path = "/src/notebooks/ID_reg/hello_world/test_img/templateF.jpg";
//     cv::Mat src_img = cv::imread(image_path);
//     cv::Mat template_img = cv::imread(templateF_path);
//     std::cout << "宽度： "<< src_img.cols << std::endl;
//     std::cout << "高度： " << src_img.rows << std::endl;
//     std::cout << "通道数： " << src_img.channels() << std::endl;

//     std::cout << "宽度： "<< template_img.cols << std::endl;
//     std::cout << "高度： " << template_img.rows << std::endl;
//     std::cout << "通道数： " << template_img.channels() << std::endl;
    
//     cv::Mat src_img2;
//     string ID_type = "reverse";
//     bool get_perspect = false;
//     src_img2 = getPerImgInF(src_img,ID_type,get_perspect);
//     std::cout << "get_perspect： "<< get_perspect << std::endl;
// //     src_img2 = imresize(src_img,640);
//     std::cout << "宽度： "<< src_img2.cols << std::endl;
//     std::cout << "高度： " << src_img2.rows << std::endl;
//     std::cout << "通道数： " << src_img2.channels() << std::endl;    
    auto sift = SIFT::create();
    cout<<CV_VERSION<<endl;
    return 0;
}




