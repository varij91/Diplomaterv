#ifndef NBODY_ALGORITHM_GPU_ALL_PAIRS_CUH
#define NBODY_ALGORITHM_GPU_ALL_PAIRS_CUH

#include "NBodyAlgorithmGPU.cuh"

__global__ void advanceKernel(float3 *pos, float *mass, float3 *acc, float eps2);

class NBodyAlgorithmGPUAllPairs : public NBodyAlgorithmGPU {

public:
    NBodyAlgorithmGPUAllPairs(std::shared_ptr<NBodyProperties> properties) : NBodyAlgorithmGPU(properties) {
        mp_mass = new float[properties->numBody];
        mp_position = new float3[properties->numBody];
        mp_velocity = new float3[properties->numBody];
        mp_acceleration = new float3[properties->numBody];
    }

    ~NBodyAlgorithmGPUAllPairs() {
        delete[] mp_mass;
        delete[] mp_position;
        delete[] mp_velocity;
        delete[] mp_acceleration;
    }

    void advance(std::vector<Body> &bodies);
    
protected:

    void unpackBodies(std::vector<Body> &bodies);
    void packBodies(std::vector<Body> &bodies);

    float  *mp_mass;
    float3 *mp_position;
    float3 *mp_velocity;
    float3 *mp_acceleration;
};

#endif // NBODY_ALGORITHM_GPU_ALL_PAIRS_CUH