#ifndef NBODY_RENDERER_H
#define NBODY_RENDERER_H

#include <memory>
#include "NBodyProperties.h"
#include "NBodySystem.h"

class NBodyRenderer {
public:
    static std::shared_ptr<NBodyProperties> mp_properties;
    static std::shared_ptr<NBodySystem> mp_system;

    void static renderCallback();
    void static idleCallback();

    void static renderMainLoop();
    void static initGL(int *argc, char *argv[]);

    void static renderSystem();

    void static getColor(unsigned int &r, unsigned int &g, unsigned int &b, int index);

    void static setProperties(const std::shared_ptr<NBodyProperties> properties) {
        mp_properties = properties;
    }
    void static setSystem(const std::shared_ptr<NBodySystem> system) {
        mp_system = system;
    }

private:
    // Tisztán statikus osztály, példány nem hozható létre
    NBodyRenderer() {}
};

#endif //NBODY_RENDERER_H