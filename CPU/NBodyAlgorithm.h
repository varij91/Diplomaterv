#ifndef NBODY_ALGORITHM_H
#define NBODY_ALGORITHM_H

#include <vector>
#include <memory>
#include "NBodyProperties.h"
#include "emmintrin.h"
#include "nmmintrin.h"

class NBodyAlgorithm {

public:
    NBodyAlgorithm() {}
    NBodyAlgorithm(std::shared_ptr<NBodyProperties> properties) : mp_properties(properties) {}

    float3 calculateAcceleration(const float3 posI, const float massJ, const float3 posJ);

    void calculateAcceleration(const float3 (&posI)[4], const float massJ, const float3 posJ, float3 (&accI)[4]);

    virtual void advance(std::vector<Body> &bodies) = 0;

protected:
    std::shared_ptr<NBodyProperties> mp_properties;
};

#endif // !NBODY_ALGORITHM_H
