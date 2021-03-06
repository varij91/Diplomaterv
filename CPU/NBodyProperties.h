#ifndef NBODY_PROPERTIES_H
#define NBODY_PROPERTIES_H
#include <vector>

enum DisplayMode {
    GUI,
    BATCH,
    PERFORMANCE
};

enum AlgorithmType {
    ALL_PAIRS,
    ALL_PAIRS_SELECTIVE,
};

enum MassInitType {
    EQUAL,
    RANDOM,
};

struct float3{
    float x;
    float y;
    float z;
    float3(float x, float y, float z)
        : x(x), y(y), z(z) {}
    float3& operator+=(const float3 &a) {
        this->x += a.x;
        this->y += a.y;
        this->z += a.z;
        return *this;
    }
};

struct Body {
    float3 position;
    float3 velocity;
    float3 acceleration;
    float mass;
    Body(float3 p, float3 v, float3 a, float m)
        : position(p), velocity(v), acceleration(a), mass(m) {}
};


struct NBodyProperties {
    DisplayMode     displayMode = GUI;
    AlgorithmType   algorithm = ALL_PAIRS;
    MassInitType    massInit = RANDOM;
    
    unsigned int    performanceRuns = 20;

    // 4-gyel oszthat� legyen (k�l�nben nem megy az SSE+OpenMP)
    unsigned int    numBody = 32;
    unsigned int    seed = 0;

    unsigned int    massScale = 100000;   // [EM]
    unsigned int    positionScale = 10;   // [AU]
    unsigned int    velocityScale = 100;   // initVelocityFactor*[AU]/[Day]
    const float     initVelocityFactor = 0.0001f;

    float           startTime = 0.0f;
    float           stepTime = 100.0f;    // [Day]
    float           endTime = 10000.0f;
    float           currentTime = 0.0f;

    bool            allowLogger = false;

    const float     gravConstant = 8.890422785943706e-10f;
    const float     eps2 = 10.0f;
    const float     velocity_dampening = 0.999f;

    //TODO IDEIGLENESEN IDE
    std::vector<unsigned int> numNeighbours;
};

#ifndef OPENMP
#define OPENMP
#endif // OPENMP

#endif // NBODY_PROPERTIES_H