######################该文件将torch模型转换成onnx以及simplify操作

import sys
sys.path.append('../../../SCENE_CLASSIFICATION_repetition/yolov5-5.0')
import argparse
import sys
import time
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.mobile_optimizer import optimize_for_mobile

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

import models
from models.common import Conv
from models.yolo import Detect
from models.experimental import attempt_load
from utils.activations import Hardswish, SiLU
from utils.general import colorstr, check_img_size, check_requirements, set_logging
from utils.torch_utils import select_device




def export_onnx(model, img, file, opset, train, dynamic,simplify):
    
    # img传入的是一个随机的矩阵即可，主要是约束模型的输入
    # file为待导出的模型的文件，并且保存相应的onnx模型的路径，ONNX: 使用的opset version为11
    # verbose 如果指定，将打印出一个导出轨道的调试描述
    # train表示训练模式下导出模型，而onnx模型用于推理，所以不需要设置为True.
    # input_names-按照顺序分配名称到图中的输入节点
    # output_names-按照顺序分配名称到图中的输出节点
    # dynamic-用于设置动态输入的参数
    # simplify-用于是否使用简化对.onnx进行简化。
    
    prefix=colorstr('ONNX:')
    try:
        check_requirements(('onnx','onnx-simplifier')) ###检查相应环境配置
        import onnx
        
        print(f'\n{prefix} starting export with onnx {onnx.__version__}...')
#         f = file.with_suffix('.onnx')
        
        torch.onnx.export(model,img,file,verbose=False,opset_version=opset,training=False,input_names=['images'],
                          output_names=['output'],dynamic_axes={'images': {0: 'batch', 2: 'height', 3: 'width'},  # shape(1,3,640,640)
                                        'output': {0: 'batch', 2: 'height',3:'width'},
                                        '471': {0: 'batch', 2: 'height',3:'width'},
                                        '491': {0: 'batch', 2: 'height',3:'width'}
                                        } if dynamic else None)
        
        model_onnx=onnx.load(file)
        onnx.checker.check_model(model_onnx)
        print("simplify:",simplify)
        if simplify: ######使用simplify简化onnx模型
            try:
                import onnxsim
                print(f'{prefix} simplifying with onnx-simplifier {onnxsim.__version__}...')
                model_onnx, check = onnxsim.simplify(
                    model_onnx,
                    input_shapes={'images': list(img.shape)} if dynamic else None)
                assert check, 'assert check failed'
                onnx.save(model_onnx,file)
            except Exception as e:
                print(f'{prefix} simplifier failure: {e}')
#         print(f'{prefix} export success, saved as {file} ({file_size(file):.1file} MB)')
#         print(f"{prefix} run --dynamic ONNX model inference with: 'python detect.py --weights {file}'")
        
        
    except Exception as e:
        print(f'\n{prefix} export failure: {e}')
        
def run(weights='',save_path='',img_size=(640,640),batch_size=1,device='cpu',include=('torchscript','onnx','coreml'),half=False,inplace=False,train=False,optimize=False,dynamic=False,grid=False,simplify=False,opset=11):
    
    
    print(grid)
    t=time.time()
    img_size *= 2 if len(img_size) == 1 else 1  # expand
    
    if simplify:
        save_path=save_path.split('.onnx')[0]+'_sim.onnx'
    
    file=Path(save_path)

    # Load PyTorch model
    device = select_device(device)
    assert not (device.type == 'cpu' and half), '--half only compatible with GPU export, i.e. use --device 0'
    model=attempt_load(weights,map_location=device)
    names=model.names
    
    # Input
    gs = int(max(model.stride))  # grid size (max stride)
    img_size = [check_img_size(x, gs) for x in img_size]  # verify img_size are gs-multiples
    
    ###########指定模型的输入尺寸,不需要真实的图像只要一个形状即可
    img = torch.zeros(batch_size, 3, *img_size).to(device)  # image size(1,3,640,640) iDetection
    
    
    # Update model
    if half:
        img, model = img.half(), model.half()  # to FP16
    model.train() if train else model.eval()  # training mode = no Detect() layer grid construction
    
    for k, m in model.named_modules():
        if isinstance(m, Conv):  # assign export-friendly activations
            if isinstance(m.act, nn.Hardswish): ####torch中nn.Hardswish和nn.SiLU激活函数在转换后需要自己定义一下，onnx转换的函数中并没有自定义
                m.act = Hardswish()
            elif isinstance(m.act, nn.SiLU):
                m.act = SiLU()
        elif isinstance(m, Detect):
            m.inplace = inplace
            m.onnx_dynamic = dynamic
            # m.forward = m.forward_export  # assign forward (optional)
            
    model.model[-1].export = not grid  # set Detect() layer grid export
    y = model(img)  # dry run
# #     import pdb
# #     pdb.set_trace()
#     ####测试模型 ,这里之所以测试一下，是因为自定义的网络层需要测试       
#     for _ in range(2):
#         y = model(img)  # dry runs
#     print(f"\n{colorstr('PyTorch:')} starting from {weights} ({file_size(weights):.1f} MB)")

    # Exports
#     if 'torchscript' in include:
#         export_torchscript(model, img, file, optimize)
    if 'onnx' in include:
        export_onnx(model, img, file, opset, train, dynamic, simplify)
#     if 'coreml' in include:
#         export_coreml(model, img, file)

    # Finish
    print(f'\nExport complete ({time.time() - t:.2f}s)'
          f"\nResults saved to {colorstr('bold', file.parent.resolve())}"
          f'\nVisualize with https://netron.app') 
    

    
def parse_opt():
    parser=argparse.ArgumentParser()
    parser.add_argument('--weights',type=str,default='/src/notebooks/SCENE_CLASSIFICATION_repetition/yolov5-5.0/runs/train/exp7/weights/best.pt',help='weights path')
    parser.add_argument('--save-path',type=str,default='/src/notebooks/c++_ID_Img_Rec_Project/torch_onnx/onnx_models/best-mtz.onnx',help='onnx path')
    parser.add_argument('--img-size',nargs='+',type=int,default=[640, 640],help='image(H,W)')
    parser.add_argument('--batch-size',type=int,default=1,help='batch size')
    parser.add_argument('--device',default='cpu',help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--half', action='store_true', help='FP16 half-precision export')
    parser.add_argument('--inplace', action='store_true', help='set YOLOv5 Detect() inplace=True')
    parser.add_argument('--train', action='store_true', help='model.train() mode')
    parser.add_argument('--optimize', action='store_true', help='TorchScript: optimize for mobile')
    parser.add_argument('--dynamic', action='store_true', help='ONNX: dynamic axes')
    parser.add_argument('--grid', action='store_true', help='export Detect() layer grid')
    parser.add_argument('--simplify', action='store_true', help='ONNX: simplify model')
    parser.add_argument('--opset', type=int, default=11, help='ONNX: opset version')
    
    opt = parser.parse_args()
    return opt


def main(opt):
    
#     import pdb
#     pdb.set_trace()
    set_logging()
    print(colorstr('export: ')+','.join(f'{k}={v}' for k,v in vars(opt).items()))
    run(**vars(opt))


if __name__=="__main__":
    opt=parse_opt()
    main(opt)