#include <stdio.h>
#include <iostream>
#include <assert.h>
#include <random>
#include <math.h>
#include "bodySystem.h"
#include "NBodyProperties.h"
#include "NBodyAlgorithmAllPairs.h"
#define GLEW_STATIC
#include <GL\glew.h>
#include <GL\freeglut.h>

int scaledvalue(unsigned int scale) {
    int sign = (rand() % 2) ? -1 : 1;
    int value = (float)(rand() % scale);
    return (sign * value);
}

void BodySystem::init() {
    assert(mp_properties->numBody > 1);
    assert(mp_properties->massScale != 0);
    assert(mp_properties->positionScale != 0);
    assert(mp_properties->velocityScale != 0);
    assert(!m_systemInitialized);

    srand(mp_properties->seed);
    float3 zeros = float3(0.0f, 0.0f, 0.0f);

    for (int i = 0; i < mp_properties->numBody;) {

        m_bodies.emplace_back(zeros, zeros, zeros, 0.0f);

        m_bodies.at(i).position.x = scaledvalue(mp_properties->positionScale);
        m_bodies.at(i).position.y = scaledvalue(mp_properties->positionScale);
        m_bodies.at(i).position.z = scaledvalue(mp_properties->positionScale);

        bool occupied = false;
        float disterr = 10e-6f;
        for (int j = 0; j < i; j++) {
            occupied = (abs(m_bodies.at(j).position.x - m_bodies.at(i).position.x) < disterr) &&
                (abs(m_bodies.at(j).position.y - m_bodies.at(i).position.y) < disterr) &&
                (abs(m_bodies.at(j).position.z - m_bodies.at(i).position.z) < disterr);
            if (occupied) break;
        }
        if (occupied) continue;

        switch (mp_properties->massInit) {
        case(EQUAL) :
            if (i == 0)
                m_bodies.at(i).mass = mp_properties->gravConstant * (float)((rand() % mp_properties->massScale) + 1.0f);
            else
                m_bodies.at(i).mass = m_bodies.at(i - 1).mass;
            break;
        case(RANDOM) :
            m_bodies.at(i).mass = mp_properties->gravConstant * (float)((rand() % mp_properties->massScale) + 1.0f);
            break;
        default :
            assert(false);
            break;
        }
        
        m_bodies.at(i).velocity.x = mp_properties->initVelocityFactor * scaledvalue(mp_properties->velocityScale);
        m_bodies.at(i).velocity.y = mp_properties->initVelocityFactor * scaledvalue(mp_properties->velocityScale);
        m_bodies.at(i).velocity.z = mp_properties->initVelocityFactor * scaledvalue(mp_properties->velocityScale);

        m_bodies.at(i).acceleration = zeros;

        i++;
    }

    m_systemInitialized = true;

}

void BodySystem::initGL(int *argc, char* argv[]) {
    glutInit(argc, argv);
    // Törlési szín beállítása
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glutInitWindowSize(650, 650);
    glutInitWindowPosition(0, 0);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
    glutCreateWindow("N-test szimuláció");
}

void BodySystem::setAlgorithm() {
    switch (mp_properties->algorithm) {
    case(ALL_PAIRS) :
        m_algorithm.reset(new NBodyAlgorithmAllPairs(mp_properties));
        m_algorithmInitialized = true;
        break;
    case(ALL_PAIRS_SELECTIVE) :
        break;
    default:
        assert(false);
        break;
    }
}

void BodySystem::integrate() {
    assert(m_systemInitialized);
    assert(m_algorithmInitialized);
    assert(mp_properties->startTime < mp_properties->endTime);

    float stepTime = mp_properties->stepTime;
    for (float i = mp_properties->startTime; i < mp_properties->endTime;) {
        m_algorithm->advance(m_bodies);

        /*std::cout << "#############################################################" << std::endl;
        std::cout << "Time: " << i << std::endl;

        for (int j = 0; j < mp_properties->numBody; j++) {
            std::cout << j << " Pos: (" << m_bodies.at(j).position.x << ", " << m_bodies.at(j).position.y << ", " << m_bodies.at(j).position.z << ")" << std::endl;
        }*/

        renderSystem(m_bodies);

        if ((i + mp_properties->stepTime) > mp_properties->endTime) {
            stepTime = mp_properties->endTime - i;
            i = mp_properties->endTime;
        }
        else {
            i += stepTime;
        }
    }

}

void BodySystem::renderSystem(std::vector<Body> bodies) {

    // Ablak törlése a korábban beállított színre
    // Törli a bitmaszknak megfelelõ buffer(eke)t
    glClear(GL_COLOR_BUFFER_BIT);
    glPointSize(7.f);
    glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
    glBegin(GL_POINTS);
    for (int i = 0; i < mp_properties->numBody; i++)
    {
        glVertex3f(bodies.at(i).position.x / (mp_properties->positionScale * 10), bodies.at(i).position.y / (mp_properties->positionScale * 10), bodies.at(i).position.z / (mp_properties->positionScale * 10));
    }
    // Done drawing points
    glEnd();
    glDisable(GL_POINT_SPRITE_ARB);
    /* Pufferek csereje, uj kep megjelenitese */
    glutSwapBuffers();
}
