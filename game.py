# 数数游戏，遇到3的倍数说3x，遇到5说5x，遇到3和5的倍数说15x
# ubuntu 2050 32s
import numpy as np
import torch
import time

begintime = time.time()
NUM_DIGITS = 16

if torch.cuda.is_available():
    CUDA = True
    print('采用GPU')
else:
    CUDA = False
    print('采用CPU')


def fizz_buzz_encode(i):
    if i % 15 == 0: return 3
    elif i % 5 == 0: return 2
    elif i % 3 == 0: return 1
    else: return 0


def fizz_buzz_decode(i, prediction):
    return [str(i), '3X', '5X', '15X'][prediction]


def binary_encode(i, num_digits):
    '''将i由10进制变为二进制，共计num_digits个位
       其中&是按位与运算符（如果相应为都相同则为1），
       >>是右移运算符（各二进制位都右移若干位）'''
    return np.array([i >> d & 1 for d in range(num_digits)][::-1]) #[::-1]的作用是对列表进行翻转

# print (binary_encode(16,14))

# 构建输入输出（输出表示类别，所以需要是一个LongTensor）
train_x = torch.Tensor(np.array(
    [binary_encode(i, NUM_DIGITS) for i in range(512, 2**NUM_DIGITS)]))
train_y = torch.LongTensor(np.array(
    [fizz_buzz_encode(i) for i in range(512, 2**NUM_DIGITS)]))

# pytorch 定义简单三层模型(输入层为10个character，输出层为4个character，中间层使用ReLU函数)
NUM_HIDDEN = 1000 # 1000好于100
model = torch.nn.Sequential(
    torch.nn.Linear(NUM_DIGITS, NUM_HIDDEN),
    torch.nn.ReLU(),

    torch.nn.Linear(
        NUM_HIDDEN,
        4)  # 4 logits,agter softmax,we get a probability distribution
)
if CUDA:
    model = model.cuda()

# 定义损失函数(这里明显是一个分类问题使用了CrossEntropyLoss function)
loss_fn = torch.nn.CrossEntropyLoss()

# 优化模型
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

# 迭代求解
BATCH_SIZE = 256  # 每128个数据截取一次
for epoch in range(1001):
    for start in range(0, len(train_x), BATCH_SIZE):

        # 数据分段
        end = start + BATCH_SIZE
        batch_x = train_x[start:end]
        batch_y = train_y[start:end]

        # 放到GPU上运行
        if CUDA:
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()

        # 训练数据
        y_pred = model(batch_x)

        # 计算损失函数
        loss = loss_fn(y_pred, batch_y)

        # print("Epoch",epoch,loss.item())

        # grad置零
        optimizer.zero_grad()

        # backward
        loss.backward()
        optimizer.step()  # gradient descent

    loss = loss_fn(y_pred, batch_y).item()
    if epoch % 100 == 0:
        print("Epoch", epoch, loss)

print(90 * '-')
print(time.time() - begintime, ' s -- 训练时间')
print(90 * '-')
begintime = time.time()

allNumberCounts=4096
test_x = torch.Tensor(np.array([binary_encode(i, NUM_DIGITS) for i in range(1, allNumberCounts)]))
if CUDA:
    test_x = test_x.cuda()
with torch.no_grad():
    test_y = model(test_x)

# zip (a,b) 组合为[(a1,b1),(a2,b2)...]
predictions = zip(range(1, 101), test_y.max(1)[1].cpu().data.tolist())
print("预测结果,前100个展示")
print([fizz_buzz_decode(i, x) for i, x in predictions])
li = [x for x in test_y.max(1)[1].cpu().data.tolist()]
print(
    "预测正确个数：",
    sum(
        test_y.max(1)[1].data.tolist() == np.array(
            [fizz_buzz_encode(i) for i in range(1, allNumberCounts)])))

print("预测错误个数：",sum(
        test_y.max(1)[1].data.tolist() != np.array(
            [fizz_buzz_encode(i) for i in range(1, allNumberCounts)])))
# 输出预测结果

i = 0
for x in li:
    i = i + 1
    if (x != fizz_buzz_encode(i)):
        print('第', i, '个 预测:', x, '，实际: ', fizz_buzz_encode(i))

print(90 * '-')
print(time.time() - begintime, ' s -- 推理时间')
print(90 * '-')
torch.save(model.state_dict(), "Game_1000.pth")
print("模型存储OK！")
torch.save(model, "Game_1000.pt")
print("模型存储OK！")


onnx_path = "onnx_model.onnx"
torch.onnx.export(model, batch_x, onnx_path,export_params=True,        # store the trained parameter weights inside the model file
                  #opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})

