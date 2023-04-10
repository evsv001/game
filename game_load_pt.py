# pt调用

import torch
import numpy as np
import time
import torchvision.transforms as transforms

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
def binary_encode(i, num_digits):
    '''将i由10进制变为二进制，共计num_digits个位
       其中&是按位与运算符（如果相应为都相同则为1），
       >>是右移运算符（各二进制位都右移若干位）'''
    return np.array([i >> d & 1 for d in range(num_digits)][::-1]) #[::-1]的作用是对列表进行翻转
def fizz_buzz_encode(i):
    if i % 15 == 0: return 3
    elif i % 5 == 0: return 2
    elif i % 3 == 0: return 1
    else: return 0
def fizz_buzz_decode(i, prediction):
    return [str(i), '3X', '5X', '15X'][prediction]


NUM_DIGITS = 16
allNumberCounts=65535
# 1 载入模型 数据----------------------------------------------------------------------
T0=time.time()
pt_path = "Game_1000.pt"
model=torch.load(pt_path)#这里已经不需要重构模型结构了，直接load就可以
test_x = torch.Tensor(np.array([binary_encode(i, NUM_DIGITS) for i in range(1, allNumberCounts)]))
# test_x = test_x.cuda()

print(f'pt   载入时间:{time.time() - T0}s')
print(90 * '_')
# 2 推理----------------------------------------------------------------------
T1=time.time()

with torch.no_grad():
    test_y = model(test_x)

print(f'pt   推理时间:{time.time() - T1}s')
print(90 * '_')

# 3 ----------------------------------------------------------------------
# zip (a,b) 组合为[(a1,b1),(a2,b2)...]
predictions = zip(range(1, 101), test_y.max(1)[1].cpu().data.tolist())
# print("预测结果,前100个展示")
# print([fizz_buzz_decode(i, x) for i, x in predictions])
# li = [x for x in test_y.max(1)[1].cpu().data.tolist()]
# print(
#     "预测正确个数：",
#     sum(
#         test_y.max(1)[1].data.tolist() == np.array(
#             [fizz_buzz_encode(i) for i in range(1, allNumberCounts)])))

print("预测错误个数：",sum(
        test_y.max(1)[1].data.tolist() != np.array(
            [fizz_buzz_encode(i) for i in range(1, allNumberCounts)])))



