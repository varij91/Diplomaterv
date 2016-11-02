#ifndef NBODY_ALGORITHM_GPU_ALL_PAIRS_CUH
#define NBODY_ALGORITHM_GPU_ALL_PAIRS_CUH

#include "NBodyAlgorithmGPU.cuh"

// Tagfüggvény nem lehet, mivel ez egy GPU-n futtatott kód
// Teszt, hogy melyik gyorsabb: NVidia szerinta float4-gyel jobb memory BW érhetõ el
__device__ float3 calculateAcceleration(const float3 posI, const float massJ, const float3 posJ, float3 accSumI);
/*__device__ float3 calculateAcceleration(const float4 posI, const float4 posMassJ, float3 accSumI);
__device__ float3 calculateAcceleration(const float *posI, const float massJ, const float *posJ, float* accSumI);*/

__device__ float3 tileCalculateAcceleration(const float3 posI, float3 accI);

__global__ void advanceKernel(float3 *g_pos, float *g_mass, float3 *g_acc);


__device__ float3 calculateAccelerationWithColor(const float3 posI, const float massJ, const float3 posJ, float3 accSumI);

__device__ float3 tileCalculateAccelerationWithColor(const float3 posI, float3 accI);

__global__ void advanceKernelWithColor(float3 *g_pos, float *g_mass, float3 *g_acc, float *g_numNeighbours);


class NBodyAlgorithmGPUAllPairs : public NBodyAlgorithmGPU {

public:
    NBodyAlgorithmGPUAllPairs(std::shared_ptr<NBodyProperties> properties) : NBodyAlgorithmGPU(properties) {}

    void advance(std::vector<Body> &bodies);

    void updateBodies(std::vector<Body> &bodies);
};

#endif // NBODY_ALGORITHM_GPU_ALL_PAIRS_CUH