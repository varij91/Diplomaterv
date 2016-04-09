#ifndef NBODY_ALGORITHM_H
#define NBODY_ALGORITHM_H

class NBodyAlgorithm {

public:
    void calculateAcceleration(const float posI[3], float accI[3], const float massJ, const float posJ[3]);

    virtual void advance(const unsigned int numBody, const float *mass,
        float *pos, float *vel, float *acc, const float timeStep) = 0;
};

#endif // !NBODY_ALGORITHM_H
