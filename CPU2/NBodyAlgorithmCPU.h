#ifndef NBODY_ALGORITHM_CPU_H
#define NBODY_ALGORITHM_CPU_H

#include "NBodyAlgorithm.h"
#include "emmintrin.h"
#include "nmmintrin.h"
#include "immintrin.h"

class NBodyAlgorithmCPU : public NBodyAlgorithm {
public:
    NBodyAlgorithmCPU(std::shared_ptr<NBodyProperties> properties) : NBodyAlgorithm(properties) {}

    float3 calculateAcceleration(const float3 posI, const float massJ, const float3 posJ);
    void calculateAcceleration(const float3(&posI)[4], const float massJ, const float3 posJ, float3(&accI)[4]);
    void calculateAcceleration(const float3(&posI)[8], const float massJ, const float3 posJ, float3(&accI)[8]);
    /*void calculateAcceleration(const float3(&posI)[4], const float massJ, const float3 posJ, float* accI);

    void calculateAcceleration(const float3(&posI)[4], const float massJ, const float3 posJ, __m128 accIx, __m128 accIy, __m128 accIz, float *accI);*/

    float3 calculateAccelerationWithColor(const float3 posI, const float massJ, const float3 posJ, unsigned int &numNeighbours);
    void calculateAccelerationWithColor(const float3(&posI)[4], const float massJ, const float3 posJ, float3(&accI)[4], unsigned int(&numNeighbours)[4]);
    void calculateAccelerationWithColor(const float3(&posI)[8], const float massJ, const float3 posJ, float3(&accI)[8], unsigned int(&numNeighbours)[8]);

    virtual void advance(std::vector<Body> &bodies) = 0;

};

#endif // NBODY_ALGORITHM_CPU_H