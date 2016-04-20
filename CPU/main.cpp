#include <math.h>

#include <iostream>
#include <fstream>
#include <assert.h>
#include <chrono>
#include <ctime>
#include <string>

#define GLEW_STATIC
#include <GL\glew.h>
#include <GL\freeglut.h>

#include "bodySystem.h"
#include "NBodyProperties.h"

using namespace std::chrono;

NBodyProperties g_properties;

static void showProgramUsage(std::string name) {
    std::cerr << "Usage: " << name << " [<option(s)> <key1=value1> <key2=value2> ...]"
        << "Options:\n"
        << "\t-h,--help\t\tShow this help message\n"
        << "Keys:\n"
        << std::endl;
}

bool commandLineParser(int argc, const char *argv[]) {
    if (argc < 2) {
        return true;
    }

    // Argumentumlista feldolgozása
    for (int i = 1; i < argc; i++) {
        std::string argument = argv[i];
        std::string key, value;
        std::size_t equalPosition = argument.find_first_of('=');

        if (equalPosition != std::string::npos) {       // Ha talált benne egyenlõségjelet
            key = argument.substr(0, equalPosition);    // kulcs = elejétõl az egyenlõségjelig
            value = argument.substr(equalPosition + 1); // érték = egyenlõségjeltõl a végéig
        }
        else {
            // Opciók (NEM kulcs érték párok feldolgozása)
            if ((argument == "-h") || (argument == "--help")) {
                showProgramUsage(argv[0]);
            }
            else {
                std::cerr << "Unknown input argument: " << argument << std::endl;
                showProgramUsage(argv[0]);
                return false;
            }
            continue;
        }

        // kulcs-érték párok feldolgozása
        if ((key == "options") || (key == "OPTIONS")) {
            // options file beolvasása
            std::ifstream file;
            file.open(value, std::ios::in);

            std::string optArgument;

            while (file >> optArgument) {
                const char *optArgv[] = { "", optArgument.c_str() };
                commandLineParser(2, optArgv);
            }
            // sorok beolvasása
            // char *dummyArgList[] = { "", currentLine };
            // commandLineParser(2, ["", currentLine]) meghívása
        }
        else if ((key == "display") || (key == "DISPLAY")) {
            if ((value == "gui") || (value == "GUI")) {
                g_properties.displayMode = DisplayMode::GUI;
            }
            else if ((value == "batch") || (value == "BATCH")) {
                g_properties.displayMode = DisplayMode::BATCH;
            }
            else {
                std::cerr << "Unknown input value for " << key << ": " << value << std::endl;
                return false;
            }
        }
        else if (key == "numBody") {
            g_properties.numBody = atoi(value.c_str());
        }
        else if (key == "alg") {
            if ((value == "all_pairs") || (value == "ALL_PAIRS")) {
                g_properties.algorithm = AlgorithmType::ALL_PAIRS;
            }
            else if ((value == "all_pairs_selective")|| (value == "ALL_PAIRS_SELECTIVE")) {
                g_properties.algorithm = AlgorithmType::ALL_PAIRS_SELECTIVE;
            }
            else {
                std::cerr << "Unknown input value for " << key << ": " << value << std::endl;
                return false;
            }
        }
        else if (key == "massScale") {
            g_properties.massScale = atoi(value.c_str());
        }
        else if (key == "posScale") {
            g_properties.positionScale = atoi(value.c_str());
        }
        else if (key == "velScale") {
            g_properties.velocityScale = atoi(value.c_str());
        }
        else if (key == "massInit") {
            if (value == "equal") {
                g_properties.massInit = MassInitType::EQUAL;
            }
            else if (value == "random") {
                g_properties.massInit = MassInitType::RANDOM;
            }
            else {
                std::cerr << "Unknown input value for " << key << ": " << value << std::endl;
                return false;
            }
        }
        else if (key == "startTime") {
            g_properties.startTime = std::stof(value);
        }
        else if (key == "stepTime") {
            g_properties.stepTime = std::stof(value);
        }
        else if (key == "endTime") {
            g_properties.endTime = std::stof(value);
        }
        else if (key == "logger") {
            if (value == "true") {
                g_properties.allowLogger = true;
            }
            else {
                g_properties.allowLogger = false;
            }
        }
        else if (key == "seed") {
            g_properties.seed = atoi(value.c_str());
        }
        else {
            std::cerr << "Unknown input argument: " << argument << std::endl;
            showProgramUsage(argv[0]);
            return false;
        }
    }

    return true;

}

int main(int argc, char* argv[])
{
    commandLineParser(argc, (const char**)argv);
    std::shared_ptr<NBodyProperties> p_properties = std::make_shared<NBodyProperties>(g_properties);
    std::unique_ptr<BodySystem> p_system = std::make_unique<BodySystem>(p_properties);

    p_system->init();
    p_system->initGL(&argc, argv);
    p_system->setAlgorithm();

    p_system->integrate();

    return 0;
}


/*int main(int argc, char* argv[])
{
    commandLineParser(argc, (const char**)argv);
    BodySystem *system = new BodySystem(&properties);
    high_resolution_clock::time_point initStart = high_resolution_clock::now();
    system->init();
    system->initGL(&argc, argv);
    system->setAlgorithm(ALL_PAIRS);

    high_resolution_clock::time_point initEnd = high_resolution_clock::now();
    auto initDuration = duration_cast<milliseconds> (initEnd - initStart).count();
    
    high_resolution_clock::time_point start = high_resolution_clock::now();
    system->integrate();
    high_resolution_clock::time_point end = high_resolution_clock::now();

    auto duration = duration_cast<milliseconds> (end - start).count();

    //float sec = duration / 10e3;
    std::cout << "Inicializacio: " << initDuration << "ms" << std::endl;
    std::cout << "Futasi ido: " << duration << "ms" << std::endl;
    //std::cout << "Ero / s: " << (NUMBODY * NUMBODY) * ((ceil((END_TIME - START_TIME)/STEP_TIME)) / (duration/1000)) << std::endl;
    return 0;
}*/