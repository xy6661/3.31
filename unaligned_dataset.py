import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import torch.backends.cudnn as cudnn
from PIL import ImageFile

cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Disable OSError: image file is truncated


class UnalignedDataset(BaseDataset):


    """"""
    """1、读取路径2、遍历路径文件夹，获取所有图片路径并将同级路径进行排序 3、获取路径长度 4、带着opt参数进入get_transform函数中只能选择进行变换"""
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.dir_A = opt.content_path
        self.dir_B = opt.style_path
        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))#make_dataset是遍历文件夹，获取所有图片路径
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        self.transform_A = get_transform(self.opt)#get_transform是根据opt中的参数智能选择进行什么样的变换
        self.transform_B = get_transform(self.opt)#get_transform是根据opt中的参数智能选择进行什么样的变换

    """ 1、如果是训练模式，index_A按顺序取出图片，index_B从B_size中随机取出图片，如果是测试模式实现两两组合
        2、"""
    # def __getitem__(self, index):
    #     if self.opt.isTrain:
    #         index_A = index#按照顺序取出图片
    #         index_B = random.randint(0, self.B_size - 1)#从bachsize中随机取出图片
    #     else:
    #         index_A = index // self.B_size
    #         index_B = index % self.B_size
    #         """index=0: index_A=0//2=0，index_B=0%2=0，组合A1和B1
    #             index=1: index_A=1//2=0，index_B=1%2=1，组合A1和B2
    #             index=2: index_A=2//2=1，index_B=2%2=0，组合A2和B1
    #             index=3: index_A=3//2=1，index_B=3%2=1，组合A2和B2
    #             index=4: index_A=4//2=2，index_B=4%2=0，组合A3和B1
    #             index=5: index_A=5//2=2，index_B=5%2=1，组合A3和B2"""
    #     A_path = self.A_paths[index_A]#从A路径中取出第index_A张图片
    #     A_img = Image.open(A_path).convert('RGB')#将A_path路径下的图片转为RGB格式
    #     A = self.transform_A(A_img)
    #     B_path = self.B_paths[index_B]
    #     B_img = Image.open(B_path).convert('RGB')
    #     B = self.transform_B(B_img)
    #
    #     name_A = os.path.basename(A_path)
    #     name_B = os.path.basename(B_path)
    #     name = name_B[:name_B.rfind('.')] + '_' + name_A[:name_A.rfind('.')] + name_A[name_A.rfind('.'):]
    #     """name_B[:name_B.rfind('.')]表示name_B文件名最后一个点之前的部分（去掉扩展名）"""
    #     #命名格式为B图片名_A图片名.扩展名
    #
    #     result = {'c': A, 's': B, 'name': name}
    #
    #     return result
    def __getitem__(self, index):
        # ... 前面的 index_A 和 index_B 的计算保持不变 ...
        if self.opt.isTrain:
            index_A = index
            index_B = random.randint(0, self.B_size - 1)
        else:
            index_A = index // self.B_size
            index_B = index % self.B_size

        # --- 修改开始 ---
        # 尝试加载内容图，如果失败则随机换一张
        while True:
            try:
                A_path = self.A_paths[index_A]
                A_img = Image.open(A_path).convert('RGB')
                A_img.load() # 确保图片数据被完整加载
                break # 加载成功，跳出循环
            except Exception as e:
                print(f"警告: 跳过损坏的内容图 {A_path}, 错误: {e}")
                index_A = random.randint(0, self.A_size - 1) # 随机选择一个新的索引

        # 尝试加载风格图，如果失败则随机换一张
        while True:
            try:
                B_path = self.B_paths[index_B]
                B_img = Image.open(B_path).convert('RGB')
                B_img.load() # 确保图片数据被完整加载
                break # 加载成功，跳出循环
            except Exception as e:
                print(f"警告: 跳过损坏的风格图 {B_path}, 错误: {e}")
                index_B = random.randint(0, self.B_size - 1) # 随机选择一个新的索引
        # --- 修改结束 ---

        A = self.transform_A(A_img)
        B = self.transform_B(B_img)

        name_A = os.path.basename(A_path)
        name_B = os.path.basename(B_path)
        name = name_B[:name_B.rfind('.')] + '_' + name_A[:name_A.rfind('.')] + name_A[name_A.rfind('.'):]

        result = {'c': A, 's': B, 'name': name, 'c_path': A_path, 's_path': B_path}
        return result

    def __len__(self):
        if self.opt.isTrain:
            return self.A_size
        else:
            return min(self.A_size * self.B_size, self.opt.num_test)