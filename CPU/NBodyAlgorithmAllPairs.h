#ifndef NBODY_ALGORITHM_ALL_PAIRS_H
#define NBODY_ALGORITHM_ALL_PAIRS_H
#include "NBodyAlgorithm.h"

class NBodyAlgorithmAllPairs : public NBodyAlgorithm {
public:
    NBodyAlgorithmAllPairs(){};

    void advance(const unsigned int numBody, const float *mass,
        float *pos, float *vel, float *acc, const float stepTime);
};
#endif