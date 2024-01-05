"""
data_utils.py
数据处理的函数,包括了
1、读取图片数据，读取相机的pose信息
2、根据传入的参数进行调整参数，主要是图片的resize参数
"""
import numpy as np
import torch 
import cv2 as cv
from pathlib import Path
from typing import Union
from enum import Enum
from tqdm import tqdm


class DataType(Enum):
    """表示读取的数据类型,是个枚举类
    """
    llff = 1
    pig = 2
    endo = 3

def __str2path(path:Union[str, Path]):
    """将str转为Path对象

    Args:
        path (Union[str, Path]): 输入的路径

    Returns:
        Path: 返回的Path对象
    """
    if isinstance(path, str):
        return Path(path)
    return path
    
def data_loader(image_path:Union[Path, str], data_type:DataType, resize_factor:float, device:torch.device = None):
    """  `数据读取的主函数`
    读取数据集，针对不同类型的数据集有不同的读取方法

    Args:
        image_path (Union[Path, str]): 表示数据集的路径
        data_type (DataType): 表示数据的类型
        resize_factor (float): 表示图片放缩的尺寸
        device (torch.device): 表示在那个设备进行
    """
    load_dict = {
        DataType.llff : _load_llff_data,
        DataType.endo: _load_endo_data,
        DataType.pig: _load_pig_data,
    }
    dict_data = load_dict[data_type](image_path, resize_factor, device)
    return dict_data
    # todo: 分为三部分进行读取
    pass
def __load_image_file(img_path:Path, device:torch.device):
    """读取目录下的图片文件, device等于输入的device设备
    Args:
        img_path (Path): 图片路径
        device (torch.device): _description_
    Returns:
        all_images[N,H,W,C]  表示所有的图片,torch,range:0-1,device:device,RGB
        N_image: 表示有多少张图片保存下来
    """
    all_img = []
    for img in tqdm(sorted(img_path.iterdir()), desc="读取路径下图片中····", leave=False):
        img_data = torch.from_numpy(cv.imread(str(img))[...,::-1].astype(np.float32) / 255.).to(device) # 转化为0-1之间的RGB图像 [H,W,C]
        all_img.append(img_data)
    return torch.stack(all_img, dim=0), len(all_img)


def __resize_image_file(ori_img_path: Path, tar_img_path:Path, resize_factor:float):
    """将ori原始目录下的image进行resize再重新保存到目标文件夹之下

    Args:
        ori_img_path (Path): 原始文件夹
        tar_img_path (Path): 目标文件夹
    """
    tar_img_path.mkdir()
    for ori_img in tqdm(ori_img_path.iterdir(), desc='resize图片中···', leave=False):  # 读取原始文件夹
        cv.imwrite(cv.resize(cv.imread(str(ori_img)), dsize=None, fx=1 / resize_factor, fy=1 / resize_factor), str(tar_img_path / f"{ori_img.name}"))
    return tar_img_path


def __load_poses_bounds(path:Path, resize_factor:int, device:torch.device):
    """将图像的pose进行处理，得到针对camear pose转换之后的结果
    c2w(pose)无需进行变换
    hwf需要处理reszie_factor
    Args:
        path (Path): poses_bounds.npy的路径位置
        resize_factor (int): 表示图像放缩的尺寸
        device (torch.device): 表示数据保存的设备
    Returns:
        dict: bounds [N,2]:表示范围, poses [N,4,4]表示相机的姿势， intrinsic [N,3,3] 表示内参  (torch.tensor)
        N:表示有多少个相机的位姿
    """
    pose_bounds = np.load(str(path)).astype(np.float32)  # 转换到float32的类型上
    bds = pose_bounds[..., -2:]  # 表示出bounds
    pose_and_hwf = np.reshape(pose_bounds[..., :-2], [-1, 3, 5])
    poses = pose_and_hwf[..., :-1]  # [N, 3, 4]
    hwf = pose_and_hwf[..., -1] / resize_factor # [N, 3] 
    bot = np.array([[[0,0,0,1]]], dtype=pose_and_hwf.dtype).repeat(poses.shape[0], axis=0)
    poses = np.concatenate([poses, bot], axis=1)  # [N, 4, 4]
    intrinsic = np.zeros([hwf.shape[0], 3, 3], dtype=pose_and_hwf.dtype)
    camera_data = {
        'bounds': torch.from_numpy(bds).to(device=device),
        'poses': torch.from_numpy(poses).to(device=device),
        'intrinsic': torch.from_numpy(intrinsic).to(device=device),
        'hwf': torch.from_numpy(hwf[0]).to(device=device)
    }
    return camera_data, pose_bounds.shape[0]


def _load_llff_data(image_path:Path, resize_factor:float, device:torch.device):
    """读取llffdata数据集

    Args:
        image_path (Path): 图片的路径(包含了图片的文件夹和poses_bounds.npy文件夹)
        resize_factor (float): 表示resize图片的尺寸
        device (torch.device): 图片、相机参数的设备位置

    Returns:
        dict: 表示总的参数位置  images, bounds, poses, intrinsic  表示了图片和边界  相机位姿 内参矩阵
    """
    image_p = image_path / f"images"
    pd = image_path / f"poses_bounds.npy"
    if resize_factor < 1:
        resize_factor = 1 / resize_factor

    if resize_factor > 1:
        image_path_resize = image_path / f"images_{resize_factor}"
    else:
        image_path_resize = image_p
    
    if image_path_resize.exists():  # 表示resize图片存在
        img, n_img = __load_image_file(image_path_resize, device=device)
    else:  # 表示resize图片不存在
        img, n_img = __load_image_file(__resize_image_file(image_p, image_path_resize, resize_factor), device=device)  # 
    
    camera_data, n_pose = __load_poses_bounds(pd, resize_factor=resize_factor, device=device)   # 表示设备的路径
    assert n_pose == n_img, f"相机姿态的个数为{n_pose}个，图片个数为{n_img}个，两者不相等"
    camera_data['images'] = img
    return camera_data
    
def _load_pig_data():
    pass

def _load_endo_data():
    pass

    