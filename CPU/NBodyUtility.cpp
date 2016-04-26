#include "NBodyUtility.h"
#include <iostream>
#include <math.h>

NBodyUtility::NBodyUtility(const std::shared_ptr<NBodyProperties> properties) {
    mp_properties = properties;
    high_resolution_clock::time_point currentTime = high_resolution_clock::now();
    m_stopwatchTime = currentTime - currentTime;
    m_numForces = properties->numBody * properties->numBody;
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
    
    unsigned int ticks = (unsigned int)ceil((mp_properties->endTime - mp_properties->startTime) / mp_properties->stepTime);
    long long int totalTime = getStopwatchTimeMilliseconds().count();
    unsigned int flop = ticks * mp_properties->numBody * mp_properties->numBody * 23 + mp_properties->numBody * 18;
    std::cout << "--------------------------------------------" << std::endl;
    std::cout << "Number of calculated forces: " << ticks * m_numForces << std::endl;
    std::cout << "Simulation ticks:            " << ticks << std::endl;
    std::cout << "Total time:                  " << totalTime << " ms" << std::endl;
    std::cout << "Forces/Second:               " << ticks * m_numForces / totalTime * 1e3 << std::endl;
    std::cout << "GFLOPS:                      " << flop / totalTime * 1e3 * 1e-9 << std::endl;
}

void NBodyUtility::printPerformace(int scale) {

    unsigned int ticks = (unsigned int)ceil((mp_properties->endTime - mp_properties->startTime) / mp_properties->stepTime);
    long long int totalTime = getStopwatchTimeMilliseconds().count();
    unsigned int flop = ticks * mp_properties->numBody * mp_properties->numBody * 23 + mp_properties->numBody * 18;
    std::cout << "--------------------------------------------" << std::endl;
    std::cout << "Number of calculated forces: " << ticks * m_numForces *scale << std::endl;
    std::cout << "Simulation ticks:            " << ticks * scale << std::endl;
    std::cout << "Total time:                  " << totalTime << " ms" << std::endl;
    std::cout << "Average time:                " << totalTime / scale << " ms" << std::endl;
    std::cout << "Forces/Second:               " << ticks * m_numForces / totalTime * 1e3 * scale << std::endl;
    std::cout << "GFLOPS:                      " << flop / totalTime * 1e3 * 1e-9 * scale << std::endl;
}