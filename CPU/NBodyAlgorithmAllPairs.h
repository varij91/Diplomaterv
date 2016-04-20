#ifndef NBODY_ALGORITHM_ALL_PAIRS_H
#define NBODY_ALGORITHM_ALL_PAIRS_H

#include "NBodyAlgorithm.h"
#include "NBodyProperties.h"

class NBodyAlgorithmAllPairs : public NBodyAlgorithm {
public:
    NBodyAlgorithmAllPairs() {};
    NBodyAlgorithmAllPairs(std::shared_ptr<NBodyProperties> properties) : NBodyAlgorithm(properties) {}

    void advance(std::vector<Body> &bodies);

};
#endif