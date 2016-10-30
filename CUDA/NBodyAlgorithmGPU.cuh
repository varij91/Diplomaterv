#ifndef NBODY_ALGORITHM_GPU_CUH
#define NBODY_ALGORITHM_GPU_CUH

#include "NBodyAlgorithm.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_functions.h"
#include "device_launch_parameters.h"

#define checkCudaError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess)
    {
        std::cerr << "CUDA failure: " << cudaGetErrorString(code) << " in " << file << " at line " << line << std::endl;
        if (abort)
            exit(code);
    }
}

// Tagfüggvény nem lehet, mivel ez egy GPU-n futtatott kód
// Teszt, hogy melyik gyorsabb: NVidia szerinta float4-gyel jobb memory BW érhetõ el
__device__ float3 calculateAcceleration(const float3 posI, const float massJ, const float3 posJ, float3 accSumI);
__device__ float3 calculateAcceleration(const float4 posI, const float4 posMassJ, float3 accSumI);
__device__ float3 calculateAcceleration(const float *posI, const float massJ, const float *posJ, float* accSumI);

__device__ float3 tileCalculateAcceleration(const float3 posI);

class NBodyAlgorithmGPU : public NBodyAlgorithm {
public:
    NBodyAlgorithmGPU(std::shared_ptr<NBodyProperties> properties) : NBodyAlgorithm(properties) {}

    virtual void advance(std::vector<Body> &bodies) = 0;

protected:


};

#endif // NBODY_ALGORITHM_GPU_CUH