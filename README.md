# 自己进行NeRF的复现

# Sturcture
- 数据读取
- NeRF网络
- 编码函数
- 训练流程
    - 光线处理
    - NDC空间转换
    - 根据weight进行重采样
    - coarse和fine进行采样
    - 输出结果保存
