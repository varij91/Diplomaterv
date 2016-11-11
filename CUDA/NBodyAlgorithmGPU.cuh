#ifndef NBODY_ALGORITHM_GPU_CUH
#define NBODY_ALGORITHM_GPU_CUH

#include "NBodyAlgorithm.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_functions.h"
#include "device_launch_parameters.h"

#define checkCudaError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        std::cerr << "CUDA failure: " << cudaGetErrorString(code) << " in " << file << " at line " << line << std::endl;
        if (abort)
            exit(code);
    }
}


class NBodyAlgorithmGPU : public NBodyAlgorithm {
public:
    NBodyAlgorithmGPU(std::shared_ptr<NBodyProperties> properties) : NBodyAlgorithm(properties) {
        mph_mass = new float[properties->numBody];
        mph_position = new float3[properties->numBody];
        mph_velocity = new float3[properties->numBody];
        mph_acceleration = new float3[properties->numBody];
        mph_numNeighbours = new float[properties->numBody];

        mph_position4 = new float4[properties->numBody];
        mph_velocity4 = new float4[properties->numBody];
    }

    ~NBodyAlgorithmGPU() {
        delete[] mph_mass;
        delete[] mph_position;
        delete[] mph_velocity;
        delete[] mph_acceleration;
        delete[] mph_numNeighbours;

        delete[] mph_position4;
        delete[] mph_velocity4;

        destroy();
    }

    virtual void advance(std::vector<Body> &bodies) = 0;

    void init(std::vector<Body> &bodies);
    void destroy();

protected:

    // Host
    float  *mph_mass;
    float3 *mph_position;
    float3 *mph_velocity;
    float3 *mph_acceleration;
    float  *mph_numNeighbours;
    
    float4 *mph_position4;
    float4 *mph_velocity4;

    // Device
    float  *mpd_mass;
    float3 *mpd_position[2];
    float3 *mpd_velocity;
    float3 *mpd_acceleration;
    float  *mpd_numNeighbours;

    float4 *mpd_position4[2];
    float4 *mpd_velocity4;

    dim3 m_gridSize;
    dim3 m_threadBlockSize;
    std::size_t m_sharedMemorySize;
    int m_writeable = 1;    // valid érték: 0, 1

    void unpackBodies(std::vector<Body> &bodies);
    void unpackBodies4(std::vector<Body> &bodies);
    void packBodies(std::vector<Body> &bodies);
    void setKernelParameters();

};

#endif // NBODY_ALGORITHM_GPU_CUH