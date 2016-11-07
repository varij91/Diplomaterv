#ifndef NBODY_UTILITY_H
#define NBODY_UTILITY_H

#include <chrono>
#include <ctime>
#include <memory>
#include "NBodyProperties.h"

using namespace std::chrono;

class NBodyUtility {

public:
    NBodyUtility(const std::shared_ptr<NBodyProperties> properties);

    // Teljesítménmérés
    void startStopwatch();
    void endStopwatch();
    void resetStopwatch();

    void calculateError(std::vector<Body> bodies1, std::vector<Body> bodies2);
    void printPerformace();
    void printPerformace(int scale);

    // Parancssor feldolgozás
    bool commandLineParse(int argc, const char *argv[]);
    void printProgramUsage(std::string name);

    high_resolution_clock::duration getStopwatchTime() {
        return m_stopwatchTime;
    }

    milliseconds getStopwatchTimeMilliseconds();

private:
    //unsigned int m_numForces = 0;

    std::shared_ptr<NBodyProperties>  mp_properties;

    high_resolution_clock::time_point m_startTime;
    high_resolution_clock::time_point m_endTime;
    high_resolution_clock::duration   m_stopwatchTime;

};

#endif //NBODY_UTILITY_H