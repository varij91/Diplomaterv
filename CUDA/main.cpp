#include <math.h>

#include <iostream>
#include <assert.h>
#include <string>
#include <vector>
#include <memory>
#include "NBodySystem.h"
#include "NBodySystemFlat.h"
#include "NBodyProperties.h"
#include "NBodyUtility.h"
#include "NBodyRenderer.h"

std::shared_ptr<NBodyProperties> gp_properties;
std::shared_ptr<NBodySystem> gp_system;
std::shared_ptr<NBodySystemFlat> gp_systemFlat;
std::shared_ptr<NBodyUtility> gp_utility;


int main(int argc, char* argv[])
{
    // Dummy rész, hogy egyelőre mindig option fájlt olvasson be
    argc = 2;
    argv[1] = "options=options.txt";

    gp_properties = std::make_shared<NBodyProperties>();
    gp_utility = std::make_shared<NBodyUtility>(gp_properties);
    gp_system = std::make_shared<NBodySystem>(gp_properties, gp_utility);

    gp_systemFlat = std::make_shared<NBodySystemFlat>(gp_properties);

    if (!gp_utility->commandLineParse(argc, (const char**)argv))
        exit(-1);

    gp_properties->currentTime = gp_properties->startTime;
    gp_properties->numNeighbours.resize(gp_properties->numBody);

    gp_system->init();
    gp_system->setAlgorithm();

#ifdef USE_FLAT_SYSTEM
    gp_systemFlat->init();
#endif

    if (gp_properties->mode == Mode::GUI) {
        NBodyRenderer::initGL(&argc, argv);
        NBodyRenderer::setProperties(gp_properties);
        NBodyRenderer::setSystem(gp_system);
        NBodyRenderer::renderMainLoop();
    }
    else if (gp_properties->mode == Mode::BATCH) {
        gp_utility->startStopwatch();
#ifdef USE_FLAT_FUNC
        gp_system->integrateFlat();
#else
#ifdef USE_FLAT_SYSTEM
        gp_systemFlat->integrate();
#else
        gp_system->integrate();
#endif
#endif
        gp_utility->endStopwatch();
        gp_utility->printPerformace();
    }
    else if (gp_properties->mode == Mode::PERFORMANCE){
        for (int i = 0; i < gp_properties->performanceRuns; i++) {
            gp_utility->startStopwatch();
#ifdef USE_FLAT_FUNC
            gp_system->integrateFlat();
#else
#ifdef USE_FLAT_SYSTEM
            gp_systemFlat->integrate();
#else
            gp_system->integrate();
#endif
#endif
            gp_utility->endStopwatch();
            
#ifdef USE_FLAT_SYSTEM
            gp_systemFlat->init();
#else
            gp_system->init();
#endif
        }
        gp_utility->printPerformace(gp_properties->performanceRuns);
    }

    return 0;
}
