#ifndef NBODY_SYSTEM_INITIALIZATOR_H
#define NBODY_SYSTEM_INITIALIZATOR_H

#include <memory>
#include <random>
#include "NBodyProperties.h"
#include "NBodyAlgorithm.h"

class NBodySystemInitializator {
public:
    static float lastMass;  // Segédváltozó azonos tömegû testek inicializálásához

    NBodySystemInitializator(std::shared_ptr<NBodyProperties> properties) {
        mp_properties = properties;
        lastMass = 0.0f;
        srand(mp_properties->seed);
    }

    float3 getNewPosition();
    float3 getNewVelocity();
    float getNewMass();

    void getNewAlgorithm(std::shared_ptr<NBodyAlgorithm> &algorithm) const;

private:
    std::shared_ptr<NBodyProperties> mp_properties;
};

#endif // NBODY_SYSTEM_INITIALIZATOR_H