#include <math.h>
#include <assert.h>
#include "NBodySystemInitializator.h"
#include "NBodyAlgorithm.h"
#include "NBodyAlgorithmCPU.h"
#include "NBodyAlgorithmCPUAllPairs.h"

float NBodySystemInitializator::lastMass;

float scaledvalue(unsigned int scale) {
    float sign = (rand() % 2) ? -1.0f : 1.0f;
    float integer = (float)(rand() % scale);
    float rmax = (float)RAND_MAX;
    float fraction = (float)(rand() % RAND_MAX) / rmax;

    return (sign * (integer + fraction));
}

float3 NBodySystemInitializator::getNewPosition() {
    float x, y, z;

    x = scaledvalue(mp_properties->positionScale);
    y = scaledvalue(mp_properties->positionScale);
    switch (mp_properties->dimension) {
    
    case Dimension::TWO:
        z = 0.0f;
        break;
    
    case Dimension::THREE:
        z = scaledvalue(mp_properties->positionScale);
        break;
    default:
        z = scaledvalue(mp_properties->positionScale);
        break;
    }

    float3 result(x, y, z);
    return result;
}

float3 NBodySystemInitializator::getNewVelocity() {
    float x, y, z;
    x = mp_properties->initVelocityFactor * scaledvalue(mp_properties->velocityScale);
    y = mp_properties->initVelocityFactor * scaledvalue(mp_properties->velocityScale);
    z = mp_properties->initVelocityFactor * scaledvalue(mp_properties->velocityScale);

    float3 result(x, y, z);
    return result;
}

float NBodySystemInitializator::getNewMass() {
    float result;
    switch (mp_properties->massInit) {
    
    case MassInitType::RANDOM:
        result = mp_properties->GRAV_CONSTANT * (float)((rand() % mp_properties->massScale) + 1.0f);
        break;
    
    case MassInitType::EQUAL:
        if (abs(lastMass) < 1e-5)   // lastMass 0.0f értékû
            result = mp_properties->GRAV_CONSTANT * (float)((rand() % mp_properties->massScale) + 1.0f);
        else
            result = lastMass;
        break;
    
    default:
        result = 0.0f;
        assert(false);
    }

    return result;
}


void NBodySystemInitializator::getNewAlgorithm(std::shared_ptr<NBodyAlgorithm> &algorithm) const {
    switch (mp_properties->algorithm) {
    case(ALL_PAIRS) :
        algorithm = std::make_shared<NBodyAlgorithmCPUAllPairs>(mp_properties);
        break;
    case(ALL_PAIRS_SELECTIVE) :
        break;
    default:
        assert(false);
        break;
    }
}