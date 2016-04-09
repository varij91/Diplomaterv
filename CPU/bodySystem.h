#ifndef BODYSYSTEM_H
#define BODYSYSTEM_H

#include "NBodyAlgorithm.h"

enum InitType {
    RANDOM,
    EQUAL
};

enum AlgorithmType {
    ALL_PAIRS,
    ALL_PAIRS_SELECTIVE
};

class BodySystem {
public:
    BodySystem(unsigned int numBodyIn) : m_numBody(numBodyIn) {}
    
    void init(unsigned int seedIn, InitType typeIn);
    void initGL(int *argc, char* argv[]);

    ~BodySystem();

    bool isSystemInitialized() {
        return m_systemInitialized;
    }

    bool isAlgorithmInitialized() {
        return m_algorithmInitialized;
    }

    void setAlgorithm(AlgorithmType typeIn);

    void integrate(float startTime, float endTime, float stepTime);
    void renderSystem(const unsigned int numBody, const float *pos);

private:
    float *m_mass;
    float *m_position;
    float *m_velocity;
    float *m_acceleration;

    NBodyAlgorithm *m_algorithm;

    unsigned int m_numBody;

    bool m_systemInitialized = false;
    bool m_algorithmInitialized = false;
};

#endif