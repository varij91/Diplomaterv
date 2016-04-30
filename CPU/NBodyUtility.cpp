#include "NBodyUtility.h"
#include <iostream>
#include <math.h>

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

void NBodyUtility::calculateError() {
    // TODO
}

void NBodyUtility::printPerformace() {
    
    unsigned long long int ticks = (unsigned long long int)ceil((mp_properties->endTime - mp_properties->startTime) / mp_properties->stepTime);
    unsigned long long int numForcesPerTicks = mp_properties->numBody * mp_properties->numBody;
    unsigned long long int numForcesTotal = numForcesPerTicks * ticks;

    long long int totalTime = getStopwatchTimeMilliseconds().count();
    unsigned long long int totalFlops = numForcesTotal * 23 + mp_properties->numBody * 18;
    std::cout << "--------------------------------------------" << std::endl;
    std::cout << "Number of calculated forces: " << ticks * mp_properties->numBody * mp_properties->numBody << std::endl;
    std::cout << "Simulation ticks:            " << ticks << std::endl;
    std::cout << "Total time:                  " << totalTime << " ms" << std::endl;
    std::cout << "Forces/Second:               " << ticks * numForcesPerTicks / totalTime * 1e3 << std::endl;
    std::cout << "GFLOPS:                      " << totalFlops / totalTime * 1e-6 << std::endl;
}

void NBodyUtility::printPerformace(int scale) {

    unsigned long long int ticks = (unsigned long long int)ceil((mp_properties->endTime - mp_properties->startTime) / mp_properties->stepTime);
    unsigned long long int numForcesPerTicks = mp_properties->numBody * mp_properties->numBody;
    unsigned long long int numForcesTotal = numForcesPerTicks * ticks;
    
    long long int totalTime = getStopwatchTimeMilliseconds().count();
    unsigned long long int totalFlops = numForcesTotal * 23 + mp_properties->numBody * 18;
    std::cout << "--------------------------------------------" << std::endl;
    std::cout << "Number of calculated forces: " << numForcesTotal *scale << std::endl;
    std::cout << "Simulation ticks:            " << ticks * scale << std::endl;
    std::cout << "Total time:                  " << totalTime << " ms" << std::endl;
    std::cout << "Average time:                " << totalTime / scale << " ms" << std::endl;
    std::cout << "Total FLOPS                  " << totalFlops << std::endl;
    std::cout << "Forces/Second:               " << numForcesTotal / totalTime * 1e3 * scale << std::endl;
    std::cout << "GFLOPS:                      " << totalFlops / totalTime * 1e-6 * scale << std::endl; // 1e3*1e-9 = 1e-6
}