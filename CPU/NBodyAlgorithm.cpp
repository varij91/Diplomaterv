#include "NBodyAlgorithm.h"
#include "NBodyProperties.h"
#include <math.h>

float3 NBodyAlgorithm::calculateAcceleration(const float3 posI, const float massJ, const float3 posJ) {
    float3 r(0.0f, 0.0f, 0.0f);
    float3 accI(0.0f, 0.0f, 0.0f);

    r.x = posJ.x - posI.x;
    r.y = posJ.y - posI.y;
    r.z = posJ.z - posI.z;

    float rabs = sqrt(r.x * r.x + r.y * r.y + r.z * r.z + mp_properties->eps2);
    float rabsInv = 1.0 / (rabs * rabs * rabs);
    float temp = massJ * rabsInv;

    // A tömegbe bele van olvasztva a G
    // Az rabsInv-be beleraktum a massJ-t, hogy ne kelljen mindig összeszorozni
    accI.x += r.x * temp;
    accI.y += r.y * temp;
    accI.z += r.z * temp;
    return accI;
}
