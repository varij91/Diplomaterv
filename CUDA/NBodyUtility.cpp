#include "NBodyUtility.h"
#include <iostream>
#include <math.h>
#include <string>
#include <fstream>
#include <assert.h>

NBodyUtility::NBodyUtility(const std::shared_ptr<NBodyProperties> properties) {
    mp_properties = properties;
    high_resolution_clock::time_point currentTime = high_resolution_clock::now();
    m_stopwatchTime = currentTime - currentTime;
    //m_numForces = properties->numBody * properties->numBody;
}

void NBodyUtility::startStopwatch() {
    m_startTime = high_resolution_clock::now();
}

void NBodyUtility::endStopwatch() {
    m_endTime = high_resolution_clock::now();
    m_stopwatchTime += m_endTime - m_startTime;
}

void NBodyUtility::resetStopwatch() {
    high_resolution_clock::time_point currentTime = high_resolution_clock::now();
    m_stopwatchTime = currentTime - currentTime;
}

milliseconds NBodyUtility::getStopwatchTimeMilliseconds() {
    duration<float> duration = m_stopwatchTime;
    return duration_cast<milliseconds>(duration);
}

void NBodyUtility::calculateError(std::vector<Body> bodies1, std::vector<Body> bodies2) {
    for (int i = 0; i < mp_properties->numBody; i++) {
        float eps = 1e-3f;
        bool hasError = false;
        float xdiff = abs(bodies1.at(i).position.x - bodies2.at(i).position.x);
        float ydiff = abs(bodies1.at(i).position.y - bodies2.at(i).position.y);
        float zdiff = abs(bodies1.at(i).position.z - bodies2.at(i).position.z);
        hasError = (xdiff > eps) || (ydiff > eps) || (zdiff > eps);
        if (hasError) {
            std::cout << i <<  " body \t(" << xdiff << ", " << ydiff << ", " << zdiff << ")" << std::endl;
            std::cout << "Failure: difference with reference model." << std::endl;
            return;
        }
    }
    std::cout << "Success: match with the reference model." << std::endl;
}

void NBodyUtility::printPerformace() {
    
    unsigned long long int ticks = (unsigned long long int)ceil((mp_properties->endTime - mp_properties->startTime) / mp_properties->stepTime);
    unsigned long long int numForcesPerTicks = mp_properties->numBody * mp_properties->numBody;
    unsigned long long int numForcesTotal = numForcesPerTicks * ticks;

    long long int totalTime = getStopwatchTimeMilliseconds().count();
    unsigned long long int totalFlops = numForcesTotal * 20 + mp_properties->numBody * 21;
    std::cout << "--------------------------------------------" << std::endl;
    std::cout << "Number of calculated forces: " << ticks * mp_properties->numBody * mp_properties->numBody << std::endl;
    std::cout << "Simulation ticks:            " << ticks << std::endl;
    std::cout << "Total time:                  " << totalTime << " ms" << std::endl;
    std::cout << "Forces/Second:               " << ticks * numForcesPerTicks / (totalTime + 1e-10f) * 1e3 << std::endl;
    std::cout << "GFLOPS:                      " << totalFlops / (totalTime + 1e-10f) * 1e-6 << std::endl;
}

void NBodyUtility::printPerformace(int scale) {

    unsigned long long int ticks = (unsigned long long int)ceil((mp_properties->endTime - mp_properties->startTime) / mp_properties->stepTime);
    unsigned long long int numForcesPerTicks = mp_properties->numBody * mp_properties->numBody;
    unsigned long long int numForcesTotal = numForcesPerTicks * ticks;
    
    float totalTime = getStopwatchTimeMilliseconds().count();
    unsigned long long int totalFlops = numForcesTotal * 23 + mp_properties->numBody * 18;
    std::cout << "--------------------------------------------" << std::endl;
    std::cout << "Number of calculated forces: " << numForcesTotal *scale << std::endl;
    std::cout << "Simulation ticks:            " << ticks * scale << std::endl;
    std::cout << "Total time:                  " << totalTime << " ms" << std::endl;
    std::cout << "Average time:                " << totalTime / scale << " ms" << std::endl;
    std::cout << "Total FLOPS                  " << totalFlops << std::endl;
    std::cout << "Forces/Second:               " << numForcesTotal / (totalTime + 1e-6f) * 1e3 * scale << std::endl;
    std::cout << "GFLOPS:                      " << totalFlops / (totalTime + 1e-6f) * 1e-6 * scale << std::endl; // 1e3*1e-9 = 1e-6
}

void NBodyUtility::printProgramUsage(std::string name) {
    std::cerr << "Usage: " << name << " [<option(s)> <key1=value1> <key2=value2> ...]"
        << "Options:\n"
        << "\t-h,--help\t\tShow this help message\n"
        << "Keys:\n"
        << std::endl;
}

bool NBodyUtility::commandLineParse(int argc, const char *argv[]) {
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
                printProgramUsage(argv[0]);
            }
            else {
                std::cerr << "Unknown input argument: " << argument << std::endl;
                printProgramUsage(argv[0]);
                return false;
            }
            continue;
        }

        // kulcs-érték párok feldolgozása
        if ((key == "options") || (key == "OPTIONS")) {
            // valueként megadott options file beolvasása
            std::ifstream file;
            file.open(value, std::ios::in);

            std::string optArgument;

            // sorok beolvasása
            while (file >> optArgument) {
                // megfelelõ argumentum típusra konverzió
                const char *optArgv[] = { "", optArgument.c_str() };
                NBodyUtility::commandLineParse(2, optArgv);  // rekurzió
            }
            // sorok beolvasása
            // char *dummyArgList[] = { "", currentLine };
            // commandLineParser(2, ["", currentLine]) meghívása
        }
        else if ((key == "display") || (key == "DISPLAY")) {
            if ((value == "gui") || (value == "GUI")) {
                mp_properties->mode = Mode::GUI;
            }
            else if ((value == "batch") || (value == "BATCH")) {
                mp_properties->mode = Mode::BATCH;
            }
            else if ((value == "perf") || (value == "PERF")) {
                mp_properties->mode = Mode::PERFORMANCE;
            }
            else {
                std::cerr << "Unknown input value for " << key << ": " << value << std::endl;
                return false;
            }
        }
        else if (key == "numBody") {
            mp_properties->numBody = atoi(value.c_str());
        }
        else if (key == "alg") {
            if ((value == "all_pairs") || (value == "ALL_PAIRS")) {
                mp_properties->algorithm = AlgorithmType::ALL_PAIRS;
            }
            else if ((value == "all_pairs_selective") || (value == "ALL_PAIRS_SELECTIVE")) {
                mp_properties->algorithm = AlgorithmType::ALL_PAIRS_SELECTIVE;
            }
            else {
                std::cerr << "Unknown input value for " << key << ": " << value << std::endl;
                return false;
            }
        }
        else if (key == "form") {
            if ((value == "SCATTER") || (value == "scatter")) {
                mp_properties->formation = BodyFormation::SCATTER;
            }
            else if ((value == "SPHERE") || (value == "sphere")) {
                mp_properties->formation = BodyFormation::SPHERE;
            }
            else {
                std::cerr << "Unknown input value for " << key << ": " << value << std::endl;
                return false;
            }
        }
        else if (key == "dimension") {
            if (value == "two") {
                mp_properties->dimension = Dimension::TWO;
            }
            else if (value == "three") {
                mp_properties->dimension = Dimension::THREE;
            }
            else {
                std::cerr << "Unknown input value for " << key << ": " << value << std::endl;
                return false;
            }
        }
        else if (key == "tech") {
            if ((value == "BASIC") || (value == "basic")) {
                mp_properties->technology = Technology::BASIC;
            }
            else if ((value == "SSE") || (value == "sse")) {
                mp_properties->technology = Technology::SSE;
            }
            else if ((value == "AVX") || (value == "avx")) {
                mp_properties->technology = Technology::AVX;
            }
            else if ((value == "GPU") || (value == "gpu")) {
                mp_properties->technology = Technology::GPU;
            }
            else {
                std::cerr << "Unknown input value for " << key << ": " << value << std::endl;
                return false;
            }
        }
        else if (key == "openmp") {
            if (value == "true") {
                mp_properties->useOpenMP = true;
            }
            else {
                mp_properties->useOpenMP = false;
            }
        }
        else if (key == "massScale") {
            mp_properties->massScale = atoi(value.c_str());
        }
        else if (key == "posScale") {
            mp_properties->positionScale = atoi(value.c_str());
        }
        else if (key == "velScale") {
            mp_properties->velocityScale = atoi(value.c_str());
        }
        else if (key == "massInit") {
            if (value == "equal") {
                mp_properties->massInit = MassInitType::EQUAL;
            }
            else if (value == "random") {
                mp_properties->massInit = MassInitType::RANDOM;
            }
            else {
                std::cerr << "Unknown input value for " << key << ": " << value << std::endl;
                return false;
            }
        }
        else if (key == "startTime") {
            mp_properties->startTime = std::stof(value);
        }
        else if (key == "stepTime") {
            mp_properties->stepTime = std::stof(value);
        }
        else if (key == "endTime") {
            mp_properties->endTime = std::stof(value);
        }
        else if (key == "logger") {
            if (value == "true") {
                mp_properties->allowLogger = true;
            }
            else {
                mp_properties->allowLogger = false;
            }
        }
        else if (key == "seed") {
            mp_properties->seed = atoi(value.c_str());
        }
        else if (key == "perfruns") {
            mp_properties->performanceRuns = atoi(value.c_str());
        }
        else if (key == "userefmod") {
            if (value == "true") {
                mp_properties->useReferenceModel = true;
            }
            else {
                mp_properties->useReferenceModel = false;
            }
        }
        else {
            std::cerr << "Unknown input argument: " << argument << std::endl;
            printProgramUsage(argv[0]);
            return false;
        }
    }

    return true;
}