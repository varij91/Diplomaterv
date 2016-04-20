#ifndef NBODY_ALGORITHM_H
#define NBODY_ALGORITHM_H

#include <vector>
#include <memory>
#include "NBodyProperties.h"

class NBodyAlgorithm {

public:
    NBodyAlgorithm() {}
    NBodyAlgorithm(std::shared_ptr<NBodyProperties> properties) : mp_properties(properties) {}

    float3 calculateAcceleration(const float3 posI, const float massJ, const float3 posJ);

    virtual void advance(std::vector<Body> &bodies) = 0;

protected:
    std::shared_ptr<NBodyProperties> mp_properties;
};

#endif // !NBODY_ALGORITHM_H
