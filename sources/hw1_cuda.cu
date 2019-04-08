//////////////////////////////////////////
// 姓名: 谢易华
// 文件说明: 输入标准差s, 计算以坐标原点为中
// 心, 边长为(6 * s + 1)的方形区域的二维高斯
// 函数的值并输出(CUDA实现)
//////////////////////////////////////////

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// 使用的Block和每个Block的Thread数量
#define BLOCK_NUM 4
#define THREAD_NUM 1024

// 计算任务常用的量
#define MAT_SIDE (6 * s + 1)
#define MAT_ELE_NUM ((6 * s + 1) * (6 * s + 1))
#define MAT_BYTES ((6 * s + 1) * (6 * s + 1) * sizeof(float))

//////////////////////////////////////////
// 函数: taskDistribute
// 函数描述: Block和Thread的任务数量分配
// 返回值: 无
// 参数描述:
// totalTaskNum: 程序总任务数
// blockTaskNum: 单个Block任务数(指针)
// threadTaskNum: 单个Thread任务数(指针)
//////////////////////////////////////////
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

//////////////////////////////////////////
// 函数: gaussian
// 函数描述: 二维高斯函数
// 返回值: 输入坐标的二维高斯函数值
// 参数描述:
// x: 横坐标x
// y: 纵坐标y
// s: 二维高斯函数的标准差
//////////////////////////////////////////
__device__ float gaussian(float x, float y, int s) {
    // 圆周率选取参考 3.141592653589793238462643383279502884
    float PI = 3.141592;
    float res = exp(-(x * x + y * y) / (2 * s * s));
    res = res / (sqrt(2 * PI) * s);
    return res;
}

//////////////////////////////////////////
// 函数: calculate
// 函数描述: 计算坐标的二维高斯函数值
// 返回值: 无
// 参数描述:
// arr: Device上的线性内存空间(指针), 用于存
// 放结果矩阵
// s: 二维高斯函数的标准差
// i: 数组位置的索引
//////////////////////////////////////////
__device__ void calculate(float *arr, int s, int i) {
    int x = i / MAT_SIDE - 3 * s;
    int y = i % MAT_SIDE - 3 * s;
    arr[i] = gaussian(x, y, s);
}

//////////////////////////////////////////
// 函数: threadMisson
// 函数描述: 每个线程的任务
// 返回值: 无
// 参数描述:
// arr: Device上的线性内存空间(指针), 用于存
// 放结果矩阵
// s: 二维高斯函数的标准差
// blockTaskNum: 单个Block任务数
// threadTaskNum: 单个Thread任务数
//////////////////////////////////////////
__global__ void threadMisson(float *arr, int s, int blockTaskNum, int threadTaskNum) {
    int bId = blockIdx.x; // Block ID
    int tId = threadIdx.x; // Thread ID
    int totalTaskNum = MAT_ELE_NUM;

    int sIdx = bId * blockTaskNum + tId * threadTaskNum;
    for (int i = sIdx; i - sIdx < threadTaskNum; i++) {
        if (bId < totalTaskNum && tId < blockTaskNum && i < (bId + 1) * blockTaskNum && i < totalTaskNum) {
            calculate(arr, s, i);
            // printf("Block %d, thread %-2d, task_id %-3d: %5.4f\n", bId, tId, i, arr[i]);
        }
    }
}

int main(int argc, char *argv[]) {
    int s = atoi(argv[1]); // StandardDeviation
    float *arr_h, *arr_d;

    // 任务分配
    int blockTaskNum, threadTaskNum;
    taskDistribute(MAT_ELE_NUM, &blockTaskNum, &threadTaskNum);

    // 内存申请
    arr_h = (float*)malloc(MAT_BYTES);
    cudaMalloc((void**)&arr_d, MAT_BYTES);

    // 计算结果矩阵, 传回Host并输出
    threadMisson<<< BLOCK_NUM, THREAD_NUM >>>(arr_d, s, blockTaskNum, threadTaskNum);
    cudaMemcpy(arr_h, arr_d, MAT_BYTES, cudaMemcpyDeviceToHost);
    for (int i = 0; i < MAT_ELE_NUM; i++)
        printf("%5.4f ", arr_h[i]);
    printf("\n");

    // 内存释放
    free(arr_h);
    cudaFree(arr_d);
    return 0;
}
