"""运行相关代码
python3 detect.py --task predict 
python3 detect.py --task train    
python3 detect.py --task export  (model为pt文件,可能需要指定batchsize )
"""

from ultralytics import RTDETR
import argparse

class RTDETRRunner:
    def __init__(self, model_path):
        """初始化RT-DETR模型
        
        Args:
            model_path (str): 模型权重文件路径
        """
        self.model = RTDETR(model_path)
        
    def predict(self, source, project='runs/detect/RT-DETR', name='predict', save=True, **kwargs):
        """执行目标检测预测
        
        Args:
            source (str): 输入源(图片/视频路径)
            project (str): 项目保存路径
            name (str): 实验名称
            save (bool): 是否保存结果
            **kwargs: 其他预测参数
        """
        results = self.model.predict(
            source=source,
            project=project,
            name=name,
            save=save,
            show=True,
            **kwargs
        )
        return results
        
    def val(self, data='coco.yaml', **kwargs):
        """验证模型性能
        
        Args:
            data (str): 数据配置文件
            **kwargs: 其他验证参数
        """
        results = self.model.val(
            data=data,
            **kwargs
        )
        return results
        
    def track(self, source, tracker='bytetrack.yaml', **kwargs):
        """执行目标跟踪
        
        Args:
            source (str): 输入视频源
            tracker (str): 跟踪器配置文件
            **kwargs: 其他跟踪参数
        """
        results = self.model.track(
            source=source,
            tracker=tracker,
            **kwargs
        )
        return results
        
    def train(self, data='coco.yaml', epochs=100, batch=16, **kwargs):
        """训练模型
        
        Args:
            data (str): 训练数据配置文件
            epochs (int): 训练轮数
            batch (int): 批次大小
            **kwargs: 其他训练参数
        """
        results = self.model.train(
            data=data,
            epochs=epochs,
            batch=batch,
            **kwargs
        )
        return results
        
    def export(self, format='onnx', **kwargs):
        """导出模型为不同格式
        
        Args:
            format (str): 导出格式 ['onnx', 'engine', 'coreml'等]
            **kwargs: 其他导出参数
        """
        results = self.model.export(
            format=format,
            **kwargs
        )
        return results

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='RT-DETR Runner')
    parser.add_argument('--model', type=str, default='runs/detect/RT-DETR/train10/weights/best.pt', help='模型路径')
    parser.add_argument('--task', type=str, default='predict', choices=['train', 'predict', 'val', 'track','export'], help='执行任务类型')
    parser.add_argument('--source', type=str, default='/home/ljq/yolov5-v7/datasets/test/videos/FW_loiter1_H20T_W.mp4', help='输入源路径(用于predict和track)')
    parser.add_argument('--data', type=str, default='ultralytics/cfg/datasets/coco128-motfly.yaml', help='数据配置文件(用于train和val)')
    parser.add_argument('--epochs', type=int, default=300, help='训练轮数(用于train)')
    parser.add_argument('--batch', type=int, default=1, help='批次大小(用于train)')
    parser.add_argument('--format', type=str, default='engine', help='onnx, engine, coreml')
    parser.add_argument('--project', type=str, default='runs/RT-DETR', help='项目保存路径')
    parser.add_argument('--name', type=str,  help='实验名称')
    args = parser.parse_args()
    if args.name is None:
        args.name = args.task
    return args

if __name__ == '__main__':
    # 解析命令行参数
    args = parse_args()
    
    # 初始化Runner
    runner = RTDETRRunner(args.model)
    
    # 根据任务类型执行相应操作
    if args.task == 'predict':
        if args.source is None:
            raise ValueError("predict任务需要指定source参数")
        results = runner.predict(
            source=args.source,
            project=args.project,
            name=args.name,
            batch=1,
        )
    
    elif args.task == 'val':
        print()
        results = runner.val(
            data=args.data,
            project=args.project,
            name=args.name,
        )
    
    elif args.task == 'export':
        results = runner.export(
            model =args.model,
            format=args.format,
            # dynamic = True,
            # batch = 4,
            # workspace = 32,
        )

