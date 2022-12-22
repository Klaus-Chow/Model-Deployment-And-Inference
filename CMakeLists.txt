cmake_minimum_required(VERSION 3.12.0)
PROJECT (c++_ID_Img_Rec_Project)

#######加载opencv
set(OpenCV_DIR "/usr/local/opencv-4.5.5/build")
find_package(OpenCV REQUIRED)
message("OPENCV FOUND")


#######加载ncnn
include_directories("./ncnn-20220420/build/install/include/ncnn")
link_directories("./ncnn-20220420/build/install/lib")
FIND_PACKAGE( OpenMP REQUIRED)  
if(OPENMP_FOUND)  
    message("OPENMP FOUND")  
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")  
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")  
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")  
endif()  

#######加载所有源代码Src
macro(ncnn_add_example name)
    if(${name} STREQUAL "main")
        # 添加所有源文件
        SET(ID_rec_pro_src
           Src/Crnn/crnn.cpp    
           Src/Yolov5/yolov5.cpp)
        ADD_EXECUTABLE(main main.cpp ${ID_rec_pro_src})#添加可执行文件
        target_link_libraries(main ncnn ${OpenCV_LIBS})
    elseif(${name} STREQUAL "hehe")
        ADD_EXECUTABLE(demo12 hehe.cpp)  
        target_link_libraries(demo12 ncnn ${OpenCV_LIBS})
        
    elseif(${name} STREQUAL "main_yolo")
        SET(ID_det_src 
           Src/Yolov5/Yolov5.cpp
           Src/Yolov5/Utils.cpp)
        ADD_EXECUTABLE(main_yolo main_yolo.cpp ${ID_det_src})
        target_link_libraries(main_yolo ncnn ${OpenCV_LIBS})
    elseif(${name} STREQUAL "main_crnn")
        SET(ID_rec_src
           Src/Crnn/Crnn.cpp
           Src/Crnn/Utils.cpp)
        ADD_EXECUTABLE(main_crnn main_crnn.cpp ${ID_rec_src})
        target_link_libraries(main_crnn ncnn ${OpenCV_LIBS})
    elseif(${name} STREQUAL "main_modelservice")
        SET(ID_modelser
           Src/Crnn/Crnn.cpp
           Src/Crnn/Utils.cpp
           Src/Yolov5/Yolov5.cpp
           Src/Yolov5/Utils.cpp
           Src/Utils/Tools.cpp
           )
        ADD_EXECUTABLE(main_modelservice main_modelservice.cpp ${ID_modelser})
        target_link_libraries(main_modelservice ncnn ${OpenCV_LIBS})
    elseif(${name} STREQUAL "main_dbnet")
        SET(bank_detext_src 
           Src/DBnet/DBnet.cpp
           Src/DBnet/Utils.cpp)
        ADD_EXECUTABLE(main_dbnet main_dbnet.cpp ${bank_detext_src})
        target_link_libraries(main_dbnet ncnn ${OpenCV_LIBS})
        
    elseif(${name} STREQUAL "main_virtual_crnn")
        SET(ID_rec_src
           Src/Crnn/Crnn2.cpp
           Src/Crnn/Utils.cpp)
        ADD_EXECUTABLE(main_virtual_crnn main_virtual_crnn.cpp ${ID_rec_src})
        target_link_libraries(main_virtual_crnn ncnn ${OpenCV_LIBS})
       
    elseif(${name} STREQUAL "main_virtual_yolov5")
        SET(yolov5_detect_src
           Src/Yolov5/yolov5_virtual.cpp
           Src/Yolov5/Utils.cpp)
        ADD_EXECUTABLE(main_virtual_yolov5 main_virtual_yolov5.cpp ${yolov5_detect_src})
        target_link_libraries(main_virtual_yolov5 ncnn ${OpenCV_LIBS})
        
    elseif(${name} STREQUAL "main_modelservice_virtual")
        SET(main_modelservice_virtual
           Src/Yolov5/yolov5_virtual.cpp
           Src/Yolov5/Utils.cpp)
        ADD_EXECUTABLE(main_modelservice_virtual main_modelservice_virtual.cpp ${main_modelservice_virtual})
        target_link_libraries(main_modelservice_virtual ncnn ${OpenCV_LIBS})    
    endif()
endmacro()


##### 将指定目录添加到编译器的头文件搜索路径之下，指定的目录被解释成当前源码路径的相对路径
include_directories (${PROJECT_SOURCE_DIR}/Include/Crnn)
include_directories (${PROJECT_SOURCE_DIR}/Include/Yolov5)
include_directories (${PROJECT_SOURCE_DIR}/Include/ModelService)
include_directories (${PROJECT_SOURCE_DIR}/Include/Utils)
include_directories (${PROJECT_SOURCE_DIR}/Include/Net_Creator)

####ncnn_add_example(main_yolo)
ncnn_add_example(main_crnn)
####ncnn_add_example(main_modelservice)
ncnn_add_example(main_virtual_crnn)
ncnn_add_example(main_virtual_yolov5)
ncnn_add_example(main_modelservice_virtual)