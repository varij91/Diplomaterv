#ifndef NBODY_ALGORITHM_GPU_ALL_PAIRS_CUH
#define NBODY_ALGORITHM_GPU_ALL_PAIRS_CUH

#include "NBodyAlgorithmGPU.cuh"

__constant__ int d_NUM_BODY, d_POSITION_SCALE;
__constant__ float d_EPS2, d_VELOCITY_DAMPENING, d_STEP_TIME;

__device__ float3 calculateAcceleration(const float3 posI, const float massJ, const float3 posJ, float3 accSumI, const int eps2);
__device__ float3 calculateAccelerationWithConst(const float3 posI, const float massJ, const float3 posJ, float3 accSumI);

__device__ float3 tileCalculateAcceleration(const float3 posI, float3 accI, const int eps2);
__device__ float3 tileCalculateAccelerationWithConst(const float3 posI, float3 accI);

__device__ float3 advance_NoSharedNoTile(float3 posI, float *g_mass, float3 *g_pos, int g_numBodies, float g_eps2);
__device__ float3 advance(float3 posI, float *g_mass, float3 *g_pos, int g_numBodies, float g_eps2);
__device__ float3 advanceWithConst(float3 posI, float *g_mass, float3 *g_pos);

__global__ void integrateKernel(float *g_mass, float3 *g_pos_old, float3 *g_pos_new, float3 *g_vel, float3 *g_acc, int g_numBodies, float g_eps2, float g_stepTime, float g_velDampening);
__global__ void integrateKernelWithConst(float *g_mass, float3 *g_posOld, float3 *g_posNew, float3 *g_vel, float3 *g_acc);

__device__ float3 calculateAccelerationWithFloat4(float4 posI, float4 posJ, float3 accSumI);

__device__ float3 advanceWithFloat4(float4 posI, float4 *g_pos);

__device__ float3 advanceWithFloat4_NoSharedNoTile(float4 posI, float4 *g_pos);

__global__ void integrateKernelWithFloat4(float4 *g_posOld, float4 *g_posNew, float4 *g_vel);


__device__ float3 calculateAccelerationWithColor(const float3 posI, const float massJ, const float3 posJ, float3 accSumI, const int eps2, float *numNeighbours, const float posScale);

__device__ float3 tileCalculateAccelerationWithColor(const float3 posI, float3 accI, const int eps2, float *numNeighbours, const float posScale);

__device__ float3 advanceWithColor(float3 posI, float *g_mass, float3 *g_pos, int g_numBodies, float g_eps2, float *numNeighbours, float g_posScale);

__global__ void integrateKernelWithColor(float *g_mass, float3 *g_posOld, float3 *g_posNew, float3 *g_vel, float3 *g_acc, int g_numBodies, float g_eps2, float g_stepTime, float g_velDampening, float *g_numNeighbours, float g_posScale);


class NBodyAlgorithmGPUAllPairs : public NBodyAlgorithmGPU {

public:
    NBodyAlgorithmGPUAllPairs(std::shared_ptr<NBodyProperties> properties) : NBodyAlgorithmGPU(properties) {}

    void advance(std::vector<Body> &bodies);

    void updateBodies(std::vector<Body> &bodies);

private: 
    bool constMemoryInitalized = false;
};

#endif // NBODY_ALGORITHM_GPU_ALL_PAIRS_CUH