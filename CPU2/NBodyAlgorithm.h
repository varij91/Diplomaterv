#ifndef NBODY_ALGORITHM_H
#define NBODY_ALGORITHM_H

#include <vector>
#include <memory>
#include "NBodyProperties.h"
#include "emmintrin.h"
#include "nmmintrin.h"

class NBodyAlgorithm {

public:
    NBodyAlgorithm(std::shared_ptr<NBodyProperties> properties) : mp_properties(properties) {}

    virtual void advance(std::vector<Body> &bodies) = 0;

protected:
    std::shared_ptr<NBodyProperties> mp_properties;
};

#endif // !NBODY_ALGORITHM_H
