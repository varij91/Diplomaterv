#ifndef NBODY_SYSTEM_FLAT_H
#define NBODY_SYSTEM_FLAT_H

#include <memory>
#include <vector>
#include "NBodySystemInitializator.h"

class NBodySystemFlat {
public:

    float *mp_mass;
    float *mp_position;
    float *mp_velocity;
    float *mp_acceleration;

    NBodySystemFlat(std::shared_ptr<NBodyProperties> properties) : mp_properties(properties) {
        mp_initializator = std::make_shared<NBodySystemInitializator>(properties);
    }

    ~NBodySystemFlat() {
        _aligned_free(mp_mass);
        _aligned_free(mp_position);
        _aligned_free(mp_velocity);
        _aligned_free(mp_acceleration);
    }

    void init();

    bool isSystemInitialized() {
        return m_systemInitialized;
    }

    void integrate();

private:
    std::shared_ptr<NBodyProperties> mp_properties;
    std::shared_ptr<NBodySystemInitializator> mp_initializator;

    bool m_systemInitialized = false;
};

#endif // NBODY_SYSTEM_FLAT_H