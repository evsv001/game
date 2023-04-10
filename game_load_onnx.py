# onnx调用,似乎cuda更加费时
import onnxruntime
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
# print(onnxruntime.get_available_providers())
# providers = ['TensorrtExecutionProvider','CUDAExecutionProvider', 'CPUExecutionProvider'] 
providers = ['CUDAExecutionProvider'] 
providers = ['CPUExecutionProvider'] 

onnx_path = "onnx_model.onnx"
ort_session = onnxruntime.InferenceSession(onnx_path,providers=providers)

# test_x = torch.Tensor([binary_encode(i, NUM_DIGITS) for i in range(1, allNumberCounts)])
test_x = [binary_encode(i, NUM_DIGITS) for i in range(1, allNumberCounts)]
# test_x = test_x.cuda()

ort_inputs = {ort_session.get_inputs()[0].name: test_x}
# ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(test_x)}
print(f'onnx 载入时间:{time.time() - T0}s')
print(90 * '_')
# 2 推理----------------------------------------------------------------------
T1 = time.time()
ort_outs = ort_session.run(None, ort_inputs)

print(f'onnx 推理时间:{time.time() - T1}s')
print(90 * '_')

# 3 显示结果----------------------------------------------------------------------
countError=0
for i in range(1,allNumberCounts):
    lis=ort_outs[0][i-1]# -1 是有道理的
    if fizz_buzz_encode(i)!=lis.argmax():
        countError=countError+1
        # print('第', i, '个 预测:', lis.argmax(), '，实际: ', fizz_buzz_encode(i))

print(f'错误：{countError}个')