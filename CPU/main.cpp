#include <math.h>
#include "defines.h"

#include <iostream>
#include <chrono>
#include <ctime>

#include <GL\glew.h>
#include <GL\freeglut.h>

#include "bodySystem.h"

using namespace std::chrono;

int main(int argc, char* argv[])
{
    BodySystem *system = new BodySystem(NUMBODY);
    high_resolution_clock::time_point initStart = high_resolution_clock::now();
    system->init(SEED, EQUAL);
    system->initGL(&argc, argv);
    system->setAlgorithm(ALL_PAIRS);

    high_resolution_clock::time_point initEnd = high_resolution_clock::now();
    auto initDuration = duration_cast<milliseconds> (initEnd - initStart).count();
    
    high_resolution_clock::time_point start = high_resolution_clock::now();
    system->integrate(START_TIME, END_TIME, STEP_TIME);
    high_resolution_clock::time_point end = high_resolution_clock::now();

    auto duration = duration_cast<milliseconds> (end - start).count();

    //float sec = duration / 10e3;
    std::cout << "Inicializacio: " << initDuration << "ms" << std::endl;
    std::cout << "Futasi ido: " << duration << "ms" << std::endl;
    std::cout << "Ero / s: " << (NUMBODY * NUMBODY) * ((ceil((END_TIME - START_TIME)/STEP_TIME)) / (duration/1000)) << std::endl;
    return 0;
}