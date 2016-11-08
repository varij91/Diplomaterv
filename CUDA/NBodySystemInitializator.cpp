#include <math.h>
#include <assert.h>

#include "NBodySystemInitializator.h"
#include "NBodyAlgorithm.h"
#include "NBodyAlgorithmCPU.h"
#include "NBodyAlgorithmCPUAllPairs.h"

#include "NBodyAlgorithmGPU.cuh"
#include "NBodyAlgorithmGPUAllPairs.cuh"

float NBodySystemInitializator::lastMass;

float scaledvalue(unsigned int scale) {
    float sign = (rand() % 2) ? -1.0f : 1.0f;
    float integer = (float)(rand() % scale);
    float rmax = (float)RAND_MAX;
    float fraction = (float)(rand() % RAND_MAX) / rmax;

    return (sign * (integer + fraction));
}

std::default_random_engine generator;
float normalvalue(float mean, float deviation) {
    std::normal_distribution<float> distribution(mean, deviation);

    return distribution(generator);
}

float3 NBodySystemInitializator::getNewPosition() {
    float x, y, z;

    x = normalvalue(0.0f, mp_properties->positionScale);
    y = normalvalue(0.0f, mp_properties->positionScale);
    switch (mp_properties->dimension) {

    case Dimension::TWO:
        z = 0.0f;
        break;

    case Dimension::THREE:
        z = normalvalue(0.0f, mp_properties->positionScale);
        break;
    default:
        z = normalvalue(0.0f, mp_properties->positionScale);
        break;
    }

    float3 result;
    result.x = x; result.y = y; result.z = z;
    return result;
}

/*float3 NBodySystemInitializator::getNewPosition() {
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

    float3 result;
    result.x = x; result.y = y; result.z = z;
    return result;
}*/

float3 NBodySystemInitializator::getNewVelocity() {
    float x, y, z;
    x = mp_properties->initVelocityFactor * normalvalue(0.0f, mp_properties->velocityScale);
    y = mp_properties->initVelocityFactor * normalvalue(0.0f, mp_properties->velocityScale);

    switch (mp_properties->dimension) {

    case Dimension::TWO:
        z = 0.0f;
        break;

    case Dimension::THREE:
        z = mp_properties->initVelocityFactor * normalvalue(0.0f, mp_properties->velocityScale);
        break;
    default:
        z = mp_properties->initVelocityFactor * normalvalue(0.0f, mp_properties->velocityScale);
        break;
    }

    float3 result;
    result.x = x; result.y = y; result.z = z;
    return result;
}

float3 NBodySystemInitializator::getNewVelocity(float3 pos) {
    // Csak a 2D van támogatva
    if (mp_properties->dimension != TWO)
        return getNewVelocity();
    
    float x, y, z;
    float dist = sqrt(pos.x * pos.x + pos.y * pos.y);
    float temp = (2*mp_properties->positionScale - dist) / mp_properties->positionScale * mp_properties->velocityScale;
    x = abs(mp_properties->initVelocityFactor * normalvalue(0.0f, mp_properties->velocityScale + temp));
    y = abs(mp_properties->initVelocityFactor * normalvalue(0.0f, mp_properties->velocityScale + temp));

    if (pos.x >= 0) {
        if (pos.y >= 0) {
            x = -x;
            y = y;
        }
        else{
            x = x;
            y = y;
        }
    }
    else {
        if (pos.y >= 0) {
            x = -x;
            y = -y;
        }
        else {
            x = x;
            y = -y;
        }
    }

    z = 0.0f;

    float3 result;
    result.x = x; result.y = y; result.z = z;
    return result;
}

float NBodySystemInitializator::getNewMass() {
    float result;
    switch (mp_properties->massInit) {
    
    case MassInitType::RANDOM:
        result = mp_properties->GRAV_CONSTANT * (abs(normalvalue(mp_properties->massScale, mp_properties->massScale)) + 1.0f);
        break;
    
    case MassInitType::EQUAL:
        if (abs(lastMass) < 1e-5)   // lastMass 0.0f értékû
            result = mp_properties->GRAV_CONSTANT * (abs(normalvalue(mp_properties->massScale, mp_properties->massScale)) + 1.0f);
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
        if (mp_properties->technology != GPU)
            algorithm = std::make_shared<NBodyAlgorithmCPUAllPairs>(mp_properties);
        else
            algorithm = std::make_shared<NBodyAlgorithmGPUAllPairs>(mp_properties);
        break;
    case(ALL_PAIRS_SELECTIVE) :
        break;
    default:
        assert(false);
        break;
    }
}