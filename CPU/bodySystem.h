#ifndef BODYSYSTEM_H
#define BODYSYSTEM_H

#include <vector>
#include <memory>

#include "NBodyAlgorithm.h"

class BodySystem {
public:
    BodySystem(std::shared_ptr<NBodyProperties> properties) : mp_properties(properties) {}

    void init();
    void initGL(int *argc, char* argv[]);

    ~BodySystem() {}

    bool isSystemInitialized() {
        return m_systemInitialized;
    }

    bool isAlgorithmInitialized() {
        return m_algorithmInitialized;
    }

    void setAlgorithm();

    void integrate();
    void renderSystem(std::vector<Body> bodies);

private:

    std::vector<Body> m_bodies;
    std::unique_ptr<NBodyAlgorithm> m_algorithm;
    std::shared_ptr<NBodyProperties> mp_properties;

    bool m_systemInitialized = false;
    bool m_algorithmInitialized = false;
};

#endif