#include <cublas_v2.h>

#define BLOCK_SIZE 16
#define TILE_SIZE 6  // Winograd F(6,3)

__global__ void winograd_transform_width_kernel(
    float *__restrict__ M,
    float *__restrict__ Y,
    const int tile_in_w,
    const int collapsed_dim_size) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int w = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx >= collapsed_dim_size || w >= tile_in_w) return;
    
    float z4 = M[0 * tile_in_w * collapsed_dim_size + w * collapsed_dim_size + idx];
    float z0 = z4;
    
    z4 = M[1 * tile_in_w * collapsed_dim_size + w * collapsed_dim_size + idx];
    z0 = z0 + z4;
    float z1 = z4;
    float z2 = z4;
    float z3 = z4;
    
    z4 = M[2 * tile_in_w * collapsed_dim_size + w * collapsed_dim_size + idx];
    z0 += z4;
    z1 += -z4;
    z2 += z4;
    z3 += -z4;
    
    z4 = M[3 * tile_in_w * collapsed_dim_size + w * collapsed_dim_size + idx];
    z0 += z4;
    z1 += 2.0f * z4;
    z2 += 4.0f * z4;
    z3 += 8.0f * z4;
    
    z4 = M[4 * tile_in_w * collapsed_dim_size + w * collapsed_dim_size + idx];
    z0 += z4;
    z1 += -2.0f * z4;
    z2 += 4.0f * z4;
    z3 += -8.0f * z4;
    
    z4 = M[5 * tile_in_w * collapsed_dim_size + w * collapsed_dim_size + idx];
    z3 += z4;
    
    Y[0 * tile_in_w * collapsed_dim_size + w * collapsed_dim_size + idx] = z0;
    Y[1 * tile_in_w * collapsed_dim_size + w * collapsed_dim_size + idx] = z1;
    Y[2 * tile_in_w * collapsed_dim_size + w * collapsed_dim_size + idx] = z2;
    Y[3 * tile_in_w * collapsed_dim_size + w * collapsed_dim_size + idx] = z3;
}

__global__ void winograd_transform_height_kernel(
    float *__restrict__ Y,
    const int tile_in_w,
    const int tile_out_h,
    const int collapsed_dim_size) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int h = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx >= collapsed_dim_size || h >= tile_out_h) return;
    
    float z4 = Y[h * tile_in_w * collapsed_dim_size + 0 * collapsed_dim_size + idx];
    float z0 = z4;
    
    z4 = Y[h * tile_in_w * collapsed_dim_size + 1 * collapsed_dim_size + idx];
    z0 += z4;
    float z1 = z4;
    float z2 = z4;
    float z3 = z4;
    
    z4 = Y[h * tile_in_w * collapsed_dim_size + 2 * collapsed_dim_size + idx];
    z0 += z4;
    z1 += -z4;
    z2 += z4;
    z3 += -z4;
    
    z4 = Y[h * tile_in_w * collapsed_dim_size + 3 * collapsed_dim_size + idx];
    z0 += z4;
    z1 += 2.0f * z4;
    z2 += 4.0f * z4;
    z3 += 8.0f * z4;
    
    z4 = Y[h * tile_in_w * collapsed_dim_size + 4 * collapsed_dim_size + idx];
    z0 += z4;
    z1 += -2.0f * z4;
    z2 += 4.0f * z4;
    z3 += -8.0f * z4;
    
    z4 = Y[h * tile_in_w * collapsed_dim_size + 5 * collapsed_dim_size + idx];
    z3 += z4;
    
    Y[h * tile_in_w * collapsed_dim_size + 0 * collapsed_dim_size + idx] = z0;
    Y[h * tile_in_w * collapsed_dim_size + 1 * collapsed_dim_size + idx] = z1;
    Y[h * tile_in_w * collapsed_dim_size + 2 * collapsed_dim_size + idx] = z2;
    Y[h * tile_in_w * collapsed_dim_size + 3 * collapsed_dim_size + idx] = z3;
}

void output_transform_cuda(float *d_M, float *d_Y, const tiling_info_t ti, const int64_t collapsed_dim_size) {
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_width((collapsed_dim_size + block.x - 1) / block.x, 
                   (ti.tile_in_w + block.y - 1) / block.y);
    
    winograd_transform_width_kernel<<<grid_width, block>>>(d_M, d_Y, ti.tile_in_w, collapsed_dim_size);
    
    dim3 grid_height((collapsed_dim_size + block.x - 1) / block.x,
                    (ti.tile_out_h + block.y - 1) / block.y);
    
    winograd_transform_height_kernel<<<grid_height, block>>>(d_Y, ti.tile_in_w, ti.tile_out_h, collapsed_dim_size);
    
    cudaDeviceSynchronize();
}
