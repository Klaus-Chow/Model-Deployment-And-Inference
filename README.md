# MDAI-Model-Deployment-And-Inference-
## 项目架构  
1. build:编译后的可执行文件  
* <font color="red">Reminds:在cmakeLists.txt里面添加要编译的文件，然后再build里面cmake ..生成编译文件，然后再make得到可执行文件  
    每修改代码，就需要重新在build下make编译一下，重新生成可执行文件  
    
        
</font>
********************

2. Include  
        -Crnn:  
           >Crnn.h:提供一个文字识别器类，其构造函数用于初始化ncnn::Net crnn，加载model.param和model.bin, 还有一个成员函数detect_crnn用于推理。
           >Utils_crnn.h:提供crnn推理中需要用到的工具函数 
        -DBNet:  
           >DBnet.h: 提供了一个文字检测器类，其构造函数用于初始化ncnn::Net dbnet,加载model.param和model.bin,还有一个成员函数detect_text用于推理。  
           >Utils_dbnet.h:提供了dbnet推理中需要用到的后处理函数
        -ModelService:  
           >ModelService.h: 提供一个modelservice构造函数,通过类的组合，添加Crnn、DBnet、Yolov5的类定义的对象作为成员变量（这样的好处就是需要用到其他算法时只要添加相应的类的对象作为成员即可）
        -Utils:  
           >Tools.h：里面提供的是串联整个项目所需要的工具函数，比如抠图、透视变换、画图、识别后对文本修正的后处理等。  
        -Yolov5:  
          >Utils_yolo.h：提供yolov5推理中需要用到的工具函数，比如生成候选框、排序、nms、IOU计算等。  
          >Yolov5.h: 提供一个目标检测器类，其构造函数用于初始化ncnn::Net yolov5，加载model.param和model.bin, 还有一个成员函数detect用于推理。
**************
        
3. Src:Include中每一个.h所对应的.cpp源文件
********

4. Output:保存中间结果

********

5. CMakeLists.txt:编译文件，自定义了函数ncnn_add_example，加载opencv、加载ncnn、加载所有源文件src、生成可执行文件并link到opencv和ncnn、include相关的头文件目录

********

6. main_crnn.cpp :用于测试crnn推理
********


7. main_dbnet.cpp :用于测试dbnet推理
********

8. main_yolo.cpp :用于测试yolo推理,yolov5中的focus自定义添加到ncnn中编译。
********

9. main_modelservice.cpp:用于身份证识别，串联了yolo模型和crnn模型  
********

10. mtzimg :用于存放int8量化所需要的校验集，这里以mtz检测业务为例  
********

11. font ：用于画图bbox时标记上中文，使用的中文字体
********

12. ncnn-20220420: ncnn框架，需要先编译，mkdir build、cd build、cmake ..、make、make install ;<font color="red">Remind：因为要将编写的c++推理文件link上ncnn，所以要make install得到lib文件夹</font>  
<font color="red">Remind:在编译ncnn时需要先安装protobuf和opencv4.5</font>  
**********  
13. onnx2ncnn_model：存放onnx转成ncnn模型，包括ncnn的fp16量化、int8量化 
*********  
            
14. torch_onnx：  
         -onnx_models:存放onnx模型  
         -yolo_export:yolo算法torch转onnx的通用代码，以及使用onnxruntime推理onnx模型的代码  
         -onnx_test_output:存放onnxruntime推理的结果图像
***************  

15. requirement.txt: 环境配置相关系数
