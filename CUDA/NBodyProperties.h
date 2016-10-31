#ifndef NBODY_PROPERTIES_H
#define NBODY_PROPERTIES_H
#include <vector>
#include "cuda_runtime.h"

enum Mode {
    GUI,
    BATCH,
    PERFORMANCE
};

enum AlgorithmType {
    ALL_PAIRS,
    ALL_PAIRS_SELECTIVE,
    BARNES_HUT
};

enum MassInitType {
    EQUAL,
    RANDOM
};

enum BodyFormation {
    SCATTER,
    SPIRAL
};

enum Dimension {
    TWO,
    THREE
};

enum Technology {
    BASIC,
    SSE,
    AVX,
    GPU
};

/*struct float3{
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
};*/

struct Body {
    float3 position;
    float3 velocity;
    float3 acceleration;
    float mass;
    Body(float3 p, float3 v, float3 a, float m)
        : position(p), velocity(v), acceleration(a), mass(m) {}
};

struct NBodyProperties {
    Mode            mode = GUI;
    AlgorithmType   algorithm = ALL_PAIRS;
    MassInitType    massInit = RANDOM;
    BodyFormation   formation = SCATTER;
    Dimension       dimension = THREE;
    Technology      technology = BASIC;

    bool            useOpenMP = false;

    unsigned int    performanceRuns = 1;

    // 4-gyel osztható legyen (különben nem megy az SSE+OpenMP)
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

    const float     GRAV_CONSTANT = 8.890422785943706e-10f;
    const float     EPS2 = 10.0f;
    const float     VELOCITY_DAMPENING = 0.999f;

    std::vector<unsigned int> numNeighbours;
};

#ifndef OPENMP
#define OPENMP
#endif // OPENMP

#ifndef USE_FLAT_FUNC
//#define USE_FLAT_FUNC
#endif // USE_FLAT_FUNC

#ifndef USE_FLAT_SYSTEM
//#define USE_FLAT_SYSTEM
#endif // USE_FLAT_SYSTEM


#endif // NBODY_PROPERTIES_H