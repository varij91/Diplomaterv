#ifndef NBODY_SYSTEM_INITIALIZATOR_H
#define NBODY_SYSTEM_INITIALIZATOR_H

#include <memory>
#include <random>
#include "NBodyProperties.h"
#include "NBodyAlgorithm.h"

class NBodySystemInitializator {
public:
    static float lastMass;  // Segédváltozó azonos tömegû testek inicializálásához

    NBodySystemInitializator(std::shared_ptr<NBodyProperties> properties);

    void init();

    float3 getNewPosition();
    float3 getNewVelocity();
    float3 getNewVelocity(float3 pos);
    float getNewMass();

    void getNewAlgorithm(std::shared_ptr<NBodyAlgorithm> &algorithm) const;

private:
    std::default_random_engine m_generator;
    std::shared_ptr<NBodyProperties> mp_properties;
    

    float3 spherePosition();
    float3 scatterPosition();
    float normalvalue(float mean, float deviation);

    int m_numCores;
    std::vector<float3> m_corePositions;
};

#endif // NBODY_SYSTEM_INITIALIZATOR_H