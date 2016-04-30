#ifndef NBODY_ALGORITHM_ALL_PAIRS_SSE_H
#define NBODY_ALGORITHM_ALL_PAIRS_SSE_H

#include "NBodyAlgorithm.h"
#include "NBodyProperties.h"

class NBodyAlgorithmAllPairsSSE : public NBodyAlgorithm {
public:
    NBodyAlgorithmAllPairsSSE() {};
    NBodyAlgorithmAllPairsSSE(std::shared_ptr<NBodyProperties> properties) : NBodyAlgorithm(properties) {}

    void advance(std::vector<Body> &bodies);

};
#endif //NBODY_ALGORITHM_ALL_PAIRS_SSE_H