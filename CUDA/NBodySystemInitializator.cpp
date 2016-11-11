#include <math.h>
#include <assert.h>

#include "NBodySystemInitializator.h"
#include "NBodyAlgorithm.h"
#include "NBodyAlgorithmCPU.h"
#include "NBodyAlgorithmCPUAllPairs.h"

#include "NBodyAlgorithmGPU.cuh"
#include "NBodyAlgorithmGPUAllPairs.cuh"

float NBodySystemInitializator::lastMass;

NBodySystemInitializator::NBodySystemInitializator(std::shared_ptr<NBodyProperties> properties) {
    mp_properties = properties;
    lastMass = 0.0f;
}

void NBodySystemInitializator::init(){
    srand(mp_properties->seed);

    if (mp_properties->numBody > 256)
        m_numCores = mp_properties->numBody / 64;
    else
        m_numCores = 1;

    for (int i = 0; i < m_numCores;) {
        float deviation = (float)mp_properties->positionScale * 5.0f;

        float3 zeros;
        zeros.x = 0.0f; zeros.y = 0.0f; zeros.z = 0.0f;

        m_corePositions.emplace_back(zeros);

        m_corePositions.at(i).x = normalvalue(0.0f, deviation);
        m_corePositions.at(i).y = normalvalue(0.0f, deviation);
        if (mp_properties->dimension == THREE)
            m_corePositions.at(i).z = normalvalue(0.0f, deviation);
        else
            m_corePositions.at(i).z = 0.0f;

        bool tooClose = false;
        float disterr = (float)mp_properties->positionScale;
        for (int j = 0; j < i; j++) {
            tooClose = (abs(m_corePositions.at(j).x - m_corePositions.at(i).x) < disterr) &&
                (abs(m_corePositions.at(j).y - m_corePositions.at(i).y) < disterr) &&
                (abs(m_corePositions.at(j).z - m_corePositions.at(i).z) < disterr);
            if (tooClose) break;
        }

        if (tooClose) continue;

        i++;
    }
}

float scaledvalue(unsigned int scale) {
    float sign = (rand() % 2) ? -1.0f : 1.0f;
    float integer = (float)(rand() % scale);
    float rmax = (float)RAND_MAX;
    float fraction = (float)(rand() % RAND_MAX) / rmax;

    return (sign * (integer + fraction));
}


float NBodySystemInitializator::normalvalue(float mean, float deviation) {
    std::normal_distribution<float> distribution(mean, deviation);

    return distribution(m_generator);
}

float3 NBodySystemInitializator::spherePosition() {
    float3 result;

    result.x = normalvalue(0.0f, mp_properties->positionScale);
    result.y = normalvalue(0.0f, mp_properties->positionScale);
    result.z = normalvalue(0.0f, mp_properties->positionScale);

    return result;
}

float3 NBodySystemInitializator::scatterPosition() {
    float3 result;
    float deviation = (float)mp_properties->positionScale / 3.0f;

    int selectedCore = rand() % m_numCores;

    result.x = normalvalue(m_corePositions.at(selectedCore).x, deviation);
    result.y = normalvalue(m_corePositions.at(selectedCore).y, deviation);
    result.z = normalvalue(m_corePositions.at(selectedCore).z, deviation);

    return result;
}

float3 NBodySystemInitializator::getNewPosition() {
    float3 result;

    switch (mp_properties->formation) {
    case BodyFormation::SCATTER:
        result = scatterPosition();
        break;
    case BodyFormation::SPHERE:
        result = spherePosition();
        break;
    default:
        result = scatterPosition();
        break;
    }

    if (mp_properties->dimension == TWO)
        result.z = 0.0f;

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