#include "NBodyAlgorithmGPU.cuh"

// szopáshegy van az eps2 átadásával (nem akarom argumentumként átadni minden egyes alkalommal
#define EPS2 10.0f

__device__ float3 tileCalculateAcceleration(const float3 posI) {
    float3 accSumI;
    extern __shared__ float3 posJ[];
    extern __shared__ float massJ[];
    for (int i = 0; i < blockDim.x; i++) {
        accSumI = calculateAcceleration(posI, massJ[i], posJ[i], accSumI);
    }
    return accSumI;
}

__device__ float3 calculateAcceleration(const float3 posI, const float massJ, const float3 posJ, float3 accSumI) {
    float3 r;

    r.x = posJ.x - posI.x;
    r.y = posJ.y - posI.y;
    r.z = posJ.z - posI.z;

    float rabs = sqrt(r.x * r.x + r.y * r.y + r.z * r.z + EPS2);
    float rabsInv = 1.0f / (rabs * rabs * rabs);
    float temp = massJ * rabsInv;

    // A tömegbe bele van olvasztva a G
    // Az rabsInv-be beleraktum a massJ-t, hogy ne kelljen mindig összeszorozni
    // Be kelle hozni az gyorulás értékek akkumulálását, mert float3/float4-gyel való mûveleteket nem támogatja a CUDA C
    // Szintén nem elhanyagolható hogy MAC mûveletet ki kell használni, nem ajánlott az akkumulálást külön elvégezni
    accSumI.x += r.x * temp;
    accSumI.y += r.y * temp;
    accSumI.z += r.z * temp;
    return accSumI;
}
