//////////////////////////////////////////
// 姓名: 谢易华
// 文件说明: 输入标准差s, 计算以坐标原点为中
// 心, 边长为(6 * s + 1)的方形区域的二维高斯
// 函数的值并输出(OpenMP实现)
//////////////////////////////////////////

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// 计算任务常用的量
#define MAT_SIDE (6 * s + 1)
#define MAT_ELE_NUM ((6 * s + 1) * (6 * s + 1))
#define MAT_BYTES ((6 * s + 1) * (6 * s + 1) * sizeof(float))

//////////////////////////////////////////
// 函数: gaussian
// 函数描述: 二维高斯函数
// 返回值: 输入坐标的二维高斯函数值
// 参数描述:
// x: 横坐标x
// y: 纵坐标y
// s: 二维高斯函数的标准差
//////////////////////////////////////////
float gaussian(float x, float y, int s) {
    float PI = 3.141592;
    float res = exp(-(x * x + y * y) / (2 * s * s));
    res = res / (sqrt(2 * PI) * s);
    return res;
}

//////////////////////////////////////////
// 函数: getArrayIdx
// 函数描述: 为了性能优化, 可以只计算 1/4的
// 矩阵, 剩下的按照中心对称直接复制, 本函数的
// 目的就是根据 i, j, dir计算数组下标
// 返回值: 对应象限(方向)的数组下标
// 参数描述:
// i: 第 i行
// j: 第 j列
// dir: 0-左上, 1-右上, 2-左下, 3-右下
// s: 二维高斯函数的标准差
//////////////////////////////////////////
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

//////////////////////////////////////////
// 函数: ompCalculate
// 函数描述: OpenMP主要执行的计算过程
// 返回值: 无
// 参数描述:
// arr: 申请的 float线性内存空间(指针), 用于
// 存放结果矩阵
// s: 二维高斯函数的标准差
//////////////////////////////////////////
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

int main(int argc, char *argv[]) {
    int s = atoi(argv[1]); // StandardDeviation
    float *arr = (float*)malloc(MAT_BYTES);

    // 多线程计算
    ompCalculate(arr, s);

    // 结果输出
    for (int i = 0; i < MAT_ELE_NUM; i++)
        printf("%5.4f ", arr[i]);
    printf("\n");

    free(arr);
    return 0;
}
