#ifndef NBODY_ALGORITHM_CPU_ALLPAIRS_H
#define NBODY_ALGORITHM_CPU_ALLPAIRS_H

#include "NBodyAlgorithmCPU.h"

class NBodyAlgorithmCPUAllPairs : public NBodyAlgorithmCPU {
public:
    NBodyAlgorithmCPUAllPairs(std::shared_ptr<NBodyProperties> properties) : NBodyAlgorithmCPU(properties) {}

    void advance(std::vector<Body> &bodies);

private:
    void advanceBasic(std::vector<Body> &bodies);
    void advanceSSE(std::vector<Body> &bodies);
    void advanceAVX(std::vector<Body> &bodies);

    inline void advanceBasicCore(std::vector<Body> &bodies, int index);
    inline void advanceBasicCoreGUI(std::vector<Body> &bodies, int index);
    inline void advanceSSECore(std::vector<Body> &bodies, int index);
    inline void advanceSSECoreGUI(std::vector<Body> &bodies, int index);
    inline void advanceAVXCore(std::vector<Body> &bodies, int index);

    inline void updateBodies(std::vector<Body> &bodies, int index, int stepTime2);
};

#endif // NBODY_ALGORITHM_CPU_ALLPAIRS_H
