#ifndef NBODY_SYSTEM_H
#define NBODY_SYSTEM_H

#include <memory>
#include <vector>
#include "NBodySystemInitializator.h"
#include "NBodyAlgorithm.h"

class NBodySystem {
public:
    std::vector<Body> m_bodies;

    NBodySystem(std::shared_ptr<NBodyProperties> properties) : mp_properties(properties) {
        mp_initializator = std::make_shared<NBodySystemInitializator>(properties);
    }

    ~NBodySystem() {}

    void init();
    void setAlgorithm();

    bool isSystemInitialized() {
        return m_systemInitialized;
    }

    bool isAlgorithmInitialized() {
        return m_algorithmInitialized;
    }

    void advance();
    void integrate();
    void integrateFlat();

private:

    std::shared_ptr<NBodyAlgorithm> mp_algorithm;
    std::shared_ptr<NBodyProperties> mp_properties;
    std::shared_ptr<NBodySystemInitializator> mp_initializator;

    bool m_systemInitialized = false;
    bool m_algorithmInitialized = false;
};

#endif // NBODY_SYSTEM_H