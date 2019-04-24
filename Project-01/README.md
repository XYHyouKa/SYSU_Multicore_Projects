# 《多核程序设计与实践》课程实验报告

## 实验内容

题目：二维高斯函数的CUDA与OpenMP实现及性能分析

要求：

- $g(x,y)=\frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{x^2+y^2}{2\sigma ^2}}$
- 输入标准差 $s=\sigma$，输出(6s+1)×(6s+1)的二维单精度浮点型数组arr
- $g(0,0)$ 位于二维数组中心 $arr[3s][3s]$
- $g(-3s,-3s)$ 位于二维数组左下角 $arr[0][0]$

## 程序设计

### 高斯函数

两种实现方式有着共通相似的部分，也就是高斯函数，独立成一个函数模块方便线程调用。由于要求精确到小数点后四位，所以float数据类型已经够用，这里的$\pi$直接选取3.141592即可，剩下的直接套用二维高斯函数公式。

```c
//////////////////////////////////////////////////////////////////////////////////
// 函数: gaussian
// 函数描述: 二维高斯函数
// 返回值: 输入坐标的二维高斯函数值
// 参数描述:
// x: 横坐标x
// y: 纵坐标y
// s: 二维高斯函数的标准差
//////////////////////////////////////////////////////////////////////////////////
float gaussian(float x, float y, int s) {
    // 圆周率选取参考 3.141592653589793238462643383279502884
    float PI = 3.141592;
    float res = exp(-(x * x + y * y) / (2 * s * s));
    res = res / (sqrt(2 * PI) * s);
    return res;
}
```

### CUDA实现

程序从main函数的argv参数中获得了s标准差，而我们要求的是以原点为中心、边长为(6 × s + 1)的方形区域内的函数值，计算量是比较直观的，主要可以有以下两种设计思路：

- 静态分配——一个线程计算一个坐标，适用于计算规模小，计算资源较多的情况
- 动态分配——根据总的计算量，将任务基本平均分配给各个Block、Thread

对于计算规模较大的情形，动态分配相当于是对静态分配的一种性能优化，具体的实现稍后解释。

任务分配需要考虑各种边界情况，总任务数可能小到比Block数量还小、也可能大到远比Block数×Thread数大得多。所以我将每个Block、Thread的任务数默认为1，如果任务数足够大，再根据具体情况修改。特别地，如果任务数不能整除块、线程数，那么各自的任务数+1。这样就会造成线程执行次数溢出，下一步我将讨论如何让线程正确执行，不会做多余的计算工作。

```C
//////////////////////////////////////////////////////////////////////////////////
// 函数: taskDistribute
// 函数描述: Block和Thread的任务数量分配
// 返回值: 无
// 参数描述:
// totalTaskNum: 程序总任务数
// blockTaskNum: 单个Block任务数(指针)
// threadTaskNum: 单个Thread任务数(指针)
//////////////////////////////////////////////////////////////////////////////////
__host__ void taskDistribute(int totalTaskNum, int *blockTaskNum, int *threadTaskNum) {
    if (totalTaskNum <= BLOCK_NUM) {
        *blockTaskNum = 1; // 总任务数比Block数量少
    } else {
        if (totalTaskNum % BLOCK_NUM == 0) // 总任务数是否整除Block数量
            *blockTaskNum = totalTaskNum / BLOCK_NUM;
        else
            *blockTaskNum = totalTaskNum / BLOCK_NUM + 1; // 不能整除时, 每个Block任务数+1
    }
    if (*blockTaskNum <= THREAD_NUM) {
        *threadTaskNum = 1; // 每个Block总任务数比Thread数量少
    } else {
        if (*blockTaskNum % THREAD_NUM == 0) // 每个Block总任务数是否整除Thread数量
            *threadTaskNum = *blockTaskNum / THREAD_NUM;
        else
            *threadTaskNum = *blockTaskNum / THREAD_NUM + 1; // 不能整除时, 每个Thread任务数+1
    }
}
```

线程执行的任务时，可以从参数中获得相应的任务分配情况，执行计算的过程会检测当前坐标是否在自己的“工作范围”内——一方面，不能计算其它线程的坐标；另一方面，不能计算超过总任务数的坐标。通过这些条件来规范线程的计算工作，代码如下：

```c
//////////////////////////////////////////////////////////////////////////////////
// 函数: threadMisson
// 函数描述: 每个线程的任务
// 返回值: 无
// 参数描述:
// arr: Device上的线性内存空间(指针), 用于存放结果矩阵
// s: 二维高斯函数的标准差
// blockTaskNum: 单个Block任务数
// threadTaskNum: 单个Thread任务数
//////////////////////////////////////////////////////////////////////////////////
__global__ void threadMisson(float *arr, int s, int blockTaskNum, int threadTaskNum) {
    int bId = blockIdx.x; // Block ID
    int tId = threadIdx.x; // Thread ID
    int totalTaskNum = MAT_ELE_NUM;

    int sIdx = bId * blockTaskNum + tId * threadTaskNum;
    for (int i = sIdx; i - sIdx < threadTaskNum; i++) {
        if (bId < totalTaskNum && tId < blockTaskNum &&
            i < (bId + 1) * blockTaskNum && i < totalTaskNum) {
            calculate(arr, s, i);
            // printf("Block %d, thread %-2d, task_id %-3d: %5.4f\n", bId, tId, i, arr[i]);
        }
    }
}
```

### OpenMP实现

结果矩阵的计算只需要两个for循环嵌套就能实现，由于循环没有上下文之类的前后依赖关系，可以直接加上并行语句。

```C
//////////////////////////////////////////////////////////////////////////////////
// 函数: ompCalculate
// 函数描述: OpenMP主要执行的计算过程
// 返回值: 无
// 参数描述:
// arr: 申请的 float线性内存空间(指针), 用于存放结果矩阵
// s: 二维高斯函数的标准差
//////////////////////////////////////////////////////////////////////////////////
void ompCalculate(float *arr, int s) {
    #pragma omp parallel for
    for (int i = 0; i < MAT_SIDE / 2 + 1; i++) {
        #pragma omp parallel for
        for (int j = 0; j < MAT_SIDE / 2 + 1; j++) {
            float res = gaussian(i - 3 * s, j - 3 * s, s);
            arr[getArrayIdx(i, j, 0, s)] = res;
            arr[getArrayIdx(i, j, 1, s)] = res;
            arr[getArrayIdx(i, j, 2, s)] = res;
            arr[getArrayIdx(i, j, 3, s)] = res;
        }
    }
}
```

可以注意到程序并没有使用全部遍历的循环，而是只遍历了矩阵的四分之一左右，这是因为二维高斯函数的对称性，所以只需要计算四分之一，剩下的只需要用坐标对称过去即可。为了简便，坐标对称使用了下面的计算函数：

```C
//////////////////////////////////////////////////////////////////////////////////
// 函数: getArrayIdx
// 函数描述: 为了性能优化, 可以只计算 1/4的矩阵, 剩下的按照中心对称直接复制, 本函数的
// 目的就是根据 i, j, dir计算数组下标
// 返回值: 对应象限(方向)的数组下标
// 参数描述:
// i: 第 i行
// j: 第 j列
// dir: 0-左上, 1-右上, 2-左下, 3-右下
// s: 二维高斯函数的标准差
//////////////////////////////////////////////////////////////////////////////////
int getArrayIdx(int i, int j, int dir, int s) {
    int index;
    switch (dir) {
        case 0: index = i * MAT_SIDE + j; break;
        case 1: index = (i + 1) * MAT_SIDE - j - 1; break;
        case 2: index = (MAT_SIDE - i - 1) * MAT_SIDE + j; break;
        case 3: index = (MAT_SIDE - i) * MAT_SIDE - j - 1; break;
        default: index = i * MAT_SIDE + j; break;
    }
    return index;
}
```

输入的i和j是原本的坐标，使用0，1，2，3表示左上，右上，左下，右下四个相对位置的对称数组下标。

## 性能分析

### CUDA

以 $s=2, 500, 1000$为样例，使用不同的Block和Thread数量进行测试，得到如下结果：

|     Block，Thread     | $s=1000$ | $s=2000$ | $s=4000$ |
| :-------------------: | :------: | :------: | :------: |
|  1 block, 16 threads  |  0.42 s  |  0.80 s  |  2.30 s  |
| 1 block, 1024 threads |  0.34 s  |  0.71 s  |  1.87 s  |
| 4 block, 1024 threads |  0.33 s  |  0.65 s  |  1.62 s  |

当Block数相同时，线程数多的时候计算会更快一些，而且成本相对等价，但是计算资源没有得到真正的利用。当多分配一些Block时，执行前的准备成本会稍微增加，但是拥有了更多的流多处理器之后，加上大量的线程，在遇到大量的计算任务时就会更快。

### OpenMP

以 $s=2, 500, 1000$的数据进行测试，结果如下：

|  $s=2$  | $s=500$ | $s=1000$ |
| :-----: | :-----: | :------: |
| 0.002 s | 0.075 s | 0.285 s  |

## 性能优化

### 块、线程任务动态分配

一个线程一个坐标计算任务其实是比较“奢侈”的，所以一般可以将任务平均分摊给各个线程，让线程得到充分利用，具体的方法在程序设计部分已经有所讨论。

### 利用对称性减小计算量

整个结果矩阵是以原点中心对称的，按照矩阵的特性，最小其实可以分割到只需要计算八分之一的矩阵，就能得到全部结果，但是从程序设计的角度上看，八分化矩阵不是很理想的选择；所以，退而求其次，可以选择四分矩阵。只需要计算四分之一的一块矩阵，在利用对称性，就可以得到全部的结果。

在这里又有了另外的问题：

- 对称化的任务交给原来负责计算的线程
- 交给另起的线程专门负责对称化

这两者哪一个效率会更高？直接来看，后者效率可能更高。但是考虑到读写冲突，负责对称化的线程需要等计算结束再进行对称，这里时间上会有额外的等待和同步开销，此处仅提供思路，具体还需要另作讨论。

