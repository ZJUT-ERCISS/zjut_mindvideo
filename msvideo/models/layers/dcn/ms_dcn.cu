#include <string.h>
#include <torch/extension.h> // 头文件引用部分
#include "ms_ext.h"
#include "deform_conv.h"
#include <stdio.h>

extern "C" int ms_deformable_conv_forward(
    int nparam,
    void** params,
    int* ndims,
    int64_t** shapes,
    const char** dtypes,
    void* stream,
    void* extra) {

    cudaStream_t custream = static_cast<cudaStream_t>(stream);
    cudaStreamSynchronize(custream);
    std::vector<int> vecp;
    vecp.push_back(0);
    vecp.push_back(1);
    vecp.push_back(2);
    vecp.push_back(3);
    vecp.push_back(4);
    vecp.push_back(5);
    auto tensors = get_torch_tensors(nparam, params, ndims, shapes, dtypes, vecp, c10::kCUDA);
    
    auto input = tensors[0];
    auto weight = tensors[1];
    auto offset = tensors[2];
    auto output = tensors[3];
    auto columns = tensors[4];
    auto ones = tensors[5];

    int *device_kw = (int *)(params[6]);
    int *device_kh = (int *)(params[7]);
    int *device_dw = (int *)(params[8]);
    int *device_dh = (int *)(params[9]);
    int *device_padw = (int *)(params[10]);
    int *device_padh = (int *)(params[11]);
    int *device_dilationw = (int *)(params[12]);
    int *device_dilationh = (int *)(params[13]);
    int *device_group = (int *)(params[14]);
    int *device_deformable_group = (int *)(params[15]);
    int *device_im2col_step = (int *)(params[16]);
    // float *output = (float *)(params[17]);

    int *host_kw = new int[1]();
    int *host_kh = new int[1]();
    int *host_dw = new int[1]();
    int *host_dh = new int[1]();
    int *host_padw = new int[1]();
    int *host_padh = new int[1]();
    int *host_dilationw = new int[1]();
    int *host_dilationh = new int[1]();
    int *host_group = new int[1]();
    int *host_deformable_group = new int[1]();
    int *host_im2col_step = new int[1]();

    cudaMemcpy(host_kw, device_kw, 1, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_kh, device_kh, 1, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_dw, device_dw, 1, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_dh, device_dh, 1, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_padw, device_padw, 1, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_padh, device_padh, 1, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_dilationw, device_dilationw, 1, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_dilationh, device_dilationh, 1, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_group, device_group, 1, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_deformable_group, device_deformable_group, 1, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_im2col_step, device_im2col_step, 1, cudaMemcpyDeviceToHost);

    int kw = *host_kw;
    int kh = *host_kh;
    int dw = *host_dw;
    int dh = *host_dh;
    int padw = *host_padw;
    int padh = *host_padh;
    int dilationw = *host_dilationw;
    int dilationh = *host_dilationh;
    int group = *host_group;
    int deformable_group = *host_deformable_group;
    int im2col_step = *host_im2col_step;

    deform_conv_forward(input,
                        weight,
                        offset,
                        output,
                        columns,
                        ones,
                        kw,
                        kh,
                        dw,
                        dh,
                        padw,
                        padh,
                        dilationw,
                        dilationh,
                        group,
                        deformable_group,
                        im2col_step);

    float *output1 = (float *)(params[3]);
    float *output2 = (float *)(params[17]);
    size_t size = sizeof(float);
    for(int i = 0; i < ndims[3]; i++){
        size *= shapes[3][i];
        // printf("%ld %ld\n", shapes[3][i], size);
    }
    
    cudaMemcpy(output2, output1, size, cudaMemcpyDeviceToDevice);
    delete[] host_kw;
    delete[] host_kh;
    delete[] host_dw;
    delete[] host_dh;
    delete[] host_padw;
    delete[] host_padh;
    delete[] host_dilationw;
    delete[] host_dilationh;
    delete[] host_group;
    delete[] host_deformable_group;
    delete[] host_im2col_step;

    return 0;
}
