#include <THC/THC.h>
#include "shift_cuda.h"
#include "cuda/shift_kernel_cuda.h"

extern THCState *state;
void shift_featuremap_cuda_forward(THCudaTensor *data, THCudaIntTensor *shift, THCudaTensor *out)
{
    THArgCheck(THCudaTensor_isContiguous(state, data), 1, "data tensor has to be contiguous");
    THArgCheck(THCudaTensor_isContiguous(state, shift), 1, "shift tensor has to be contiguous");

    int batch_size = THCudaTensor_size(state, data, 0);
    int channels = THCudaTensor_size(state, data, 2);
    int tsize = THCudaTensor_size(state, data, 1);
    int hwsize = THCudaTensor_size(state, data, 3);
    int groupsize = THCudaTensor_size(state, shift, 1);

    ShiftDataCudaForward(THCState_getCurrentStream(state),
                        THCudaTensor_data(state, data),
                        THCudaIntTensor_data(state, shift),
                        batch_size,
                        channels,
                        tsize,
                        hwsize,
                        groupsize,
                        THCudaTensor_data(state, out));
}

void shift_featuremap_cuda_backward(THCudaTensor *grad_output, THCudaIntTensor *shift, THCudaTensor *grad_input)
{
    THArgCheck(THCudaTensor_isContiguous(state, grad_output), 1, "data tensor has to be contiguous");
    THArgCheck(THCudaTensor_isContiguous(state, shift), 1, "shift tensor has to be contiguous");

    int batch_size = THCudaTensor_size(state, grad_output, 0);
    int channels = THCudaTensor_size(state, grad_output, 2);
    int tsize = THCudaTensor_size(state, grad_output, 1);
    int hwsize = THCudaTensor_size(state, grad_output, 3);
    int groupsize = THCudaTensor_size(state, shift, 1);

    ShiftDataCudaBackward(THCState_getCurrentStream(state),
                        THCudaTensor_data(state, grad_output),
                        THCudaIntTensor_data(state, shift),
                        batch_size,
                        channels,
                        tsize,
                        hwsize,
                        groupsize,
                        THCudaTensor_data(state, grad_input));
}
