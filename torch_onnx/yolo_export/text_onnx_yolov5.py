import onnx
import torch
import torch as t
import cv2
from PIL import ImageFont
import time
import torchvision
import onnxruntime
import random
import os
import glob
import numpy as np

font_path='../../font/e.ttf'
font=ImageFont.truetype(font_path,20)


def make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, t.Tensor) else np.copy(x)
    
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

####这一部分针对的是动态尺寸变换，
def letterbox_dynamic(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


def non_max_supperssion(prediction,conf_thres=0.25,iou_thres=0.45,classes=None,agnostic=False,multi_label=False,labels=(),max_det=300):
    
    
    nc=prediction.shape[2]-5
    xc=prediction[...,4]>conf_thres
    
    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'
    
    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS
    t_start=time.time()
    output = [t.zeros((0, 6), device=prediction.device)] * prediction.shape[0]

    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = t.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = t.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = t.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = t.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == t.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t_start) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output

def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2
    
def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = max(img1_shape) / max(img0_shape)  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords

def plot_one_box(x, img, color=None, label=None, line_thickness=None,font=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
#     if label:
#         if (isinstance(img,np.ndarray)):  ##如果是opencv类型,使用PIL里面的ImageDraw画图的话就需要转换一下。
#             img=Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
#         draw=ImageDraw.Draw(img)
#         # 绘制文本
#         draw.text((c1[0], c1[1] - 50), label, (255,0,0), font=font)
#         # 转换回OpenCV格式
#     return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

    
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        
#         import pdb
#         pdb.set_trace()
        cv2.rectangle(img, c1, c2, color, 1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
#         img=Image.fromarray(img)####使用PIL
#         draw = ImageDraw.Draw(img)
#         draw.text((c1[0], c1[1] - 50), label, fill=(255,0,0), font=font) 
#         # 第一个参数为打印的坐标，第二个为打印的文本，第三个为字体颜色，第四个为字体
#         img = np.array(img)[:,:,::-1]
#     return img

#####使用onnxruntime对转换得到的onnx模型进行验证
######测试/src/notebooks/YOLOV5/offiec/test/images中的106张的推理速度以及准确性


###输入图像
image_path="/src/notebooks/YOLOV5/offiec/test/images/15520334000.jpg"
# image_path="13834796639.jpg"
# image_path="./build/bus.jpg"
img_savepath=image_path.split('/')[-1]
img0=cv2.imread(image_path)
im=cv2.imread(image_path).astype(np.float32)
im_d,r,(dw,dh)=letterbox_dynamic(im)
img=im_d.reshape(1,im_d.shape[0],im_d.shape[1],im_d.shape[2])
print(img.shape)
img=np.stack(img,0)
print(img.shape)
img=img[...,::-1]
##BHWC----->BCHW
img=img.transpose((0,3,1,2))
img=np.ascontiguousarray(img)
print(img.shape)
img=img/255.0
if len(img.shape)==3:
    img=img[None] ###expand for batch dim
    
    
####加载模式
onnx_model=onnxruntime.InferenceSession("../onnx_models/yolov5/best_dynamic.onnx")
    
ort_inputs = {onnx_model.get_inputs()[0].name: img}
ort_outs=onnx_model.run(None,ort_inputs)
####对于动态输出，只能先输出三个map，--grid不能设置，因为组合在一起，而dynamic时，四个输出都是用dynamic并不能成功
grid = [torch.zeros(1)] * 3
ort_out0=torch.tensor(ort_outs[0])
ort_out1=torch.tensor(ort_outs[1])
ort_out2=torch.tensor(ort_outs[2])

grid[0]=make_grid(ort_out0.shape[3],ort_out0.shape[2]).to(ort_out0)
grid[1]=make_grid(ort_out1.shape[3],ort_out1.shape[2]).to(ort_out1)
grid[2]=make_grid(ort_out2.shape[3],ort_out2.shape[2]).to(ort_out2)

y0=ort_out0.sigmoid()
y1=ort_out1.sigmoid()
y2=ort_out2.sigmoid()

anchor_grid=[]
z=[]
anchor_grid.append(np.array([[10.,13.],[16.,30.],[33.,23.]]).reshape(1,3,1,1,2))
anchor_grid.append(np.array([[30.,61.],[62.,45.],[59.,119.]]).reshape(1,3,1,1,2))
anchor_grid.append(np.array([[116.,90.],[156.,198.],[373.,326.]]).reshape(1,3,1,1,2))


######根据偏移量在grid上进行回归，注意这只是只是在feature map的grid上，后处理还要仿射到原图上
######根据偏移量在grid上进行回归，注意这只是只是在feature map的grid上，后处理还要仿射到原图上
y0[..., 0:2] = (y0[..., 0:2] * 2. - 0.5 + grid[0]) * 8  # xy
y0[..., 2:4] = (y0[..., 2:4] * 2) ** 2 * anchor_grid[0]  # wh
y1[..., 0:2] = (y1[..., 0:2] * 2. - 0.5 + grid[1]) * 16  # xy
y1[..., 2:4] = (y1[..., 2:4] * 2) ** 2 * anchor_grid[1]  # wh
y2[..., 0:2] = (y2[..., 0:2] * 2. - 0.5 + grid[2]) * 32  # xy
y2[..., 2:4] = (y2[..., 2:4] * 2) ** 2 * anchor_grid[2]  # wh

####合并三个feature map
z.append(y0.view(ort_out0.shape[0],-1,6))
z.append(y1.view(ort_out1.shape[0],-1,6))
z.append(y2.view(ort_out2.shape[0],-1,6))
z=torch.cat(z,1)


####得到模型的三个输出后就是后处理部分
pred=non_max_supperssion(z,0.25,0.45,None,False,max_det=1000)




names=['MTZ']
for i,det in enumerate(pred):
    print(det[:,:4])
    im0s=img0.copy() ###获取原始图像
    if det is not None and len(det):
        
        det[:,:4]=scale_coords(img.shape[2:],det[:,:4],im0s.shape).round()
        print(det[:,:4])
        #Rescale boxes from img_size to im0 size,将得到bbox坐标转换到原始图像上。img是resize+padding后的图，
#         im0s是原始图
        for c in det[:,-1].unique():
            n=(det[:,-1]==c).sum()
        for *xyxy,conf,cls in det:
            label='%s %.2f'%(names[int(cls)],conf)
            plot_one_box(xyxy, im0s, label=label, color=None, line_thickness=3,font=font)
cv2.imwrite('./result_'+img_savepath,im0s)