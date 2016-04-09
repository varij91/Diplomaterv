#include <stdio.h>
#include <iostream>
#include <assert.h>
#include <random>
#include <math.h>
#include "bodySystem.h"
#include "defines.h"
#include "NBodyAlgorithmAllPairs.h"
#include <GL\glew.h>
#include <GL\freeglut.h>

void BodySystem::init(unsigned int seedIn, InitType typeIn) {
    
    assert(m_numBody > 1);
    assert(!m_systemInitialized);

    srand(seedIn);

    m_mass          = new float[m_numBody];
    m_position      = new float[3 * m_numBody];
    m_velocity      = new float[3 * m_numBody];
    m_acceleration  = new float[3 * m_numBody];

    switch (typeIn)
    {
    case(EQUAL) :
        for (int i = 0; i < m_numBody;) {

            float sign = (rand() % 2) ? -1.0f : 1.0f;
            m_position[3 * i]     = sign * (float)(rand() % UNIVERSE_SCALE);
            sign = (rand() % 2) ? -1.0f : 1.0f;
            m_position[3 * i + 1] = sign * (float)(rand() % UNIVERSE_SCALE);
            sign = (rand() % 2) ? -1.0f : 1.0f;
            m_position[3 * i + 2] = sign * (float)(rand() % UNIVERSE_SCALE);

            bool occupied = false;
            float disterr = 10e-6f;
            for (int j = 0; j < i; j++) {
                occupied = (abs(m_position[3 * j] - m_position[3 * i]) < disterr) &&
                    (abs(m_position[3 * j + 1] - m_position[3 * i + 1]) < disterr) &&
                    (abs(m_position[3 * j + 2] - m_position[3 * i + 2]) < disterr);
                if (occupied) break;
            }
            if (occupied) continue;

            if (i == 0)
                m_mass[i] = G * (float)((rand() % MASS_SCALE) + 1);
            else
                m_mass[i] = m_mass[i - 1];

            sign = (rand() % 2) ? -1.0f : 1.0f;
            m_velocity[3 * i] = (float)((rand() % (VELOCITY_SCALE)) / VELOCITY_SCALE);
            sign = (rand() % 2) ? -1.0f : 1.0f;
            m_velocity[3 * i + 1] = (float)((rand() % (VELOCITY_SCALE)) / VELOCITY_SCALE);
            sign = (rand() % 2) ? -1.0f : 1.0f;
            m_velocity[3 * i + 2] = (float)((rand() % (VELOCITY_SCALE)) / VELOCITY_SCALE);

            m_acceleration[3 * i] = 0.0f;
            m_acceleration[3 * i + 1] = 0.0f;
            m_acceleration[3 * i + 2] = 0.0f;

            i++;
        }
        break;
    case(RANDOM):
        for (int i = 0; i < m_numBody;) {

            float sign = (rand() % 2) ? -1.0f : 1.0f;
            m_position[3 * i] = sign * (float)(rand() % UNIVERSE_SCALE);
            sign = (rand() % 2) ? -1.0f : 1.0f;
            m_position[3 * i + 1] = sign * (float)(rand() % UNIVERSE_SCALE);
            sign = (rand() % 2) ? -1.0f : 1.0f;
            m_position[3 * i + 2] = sign * (float)(rand() % UNIVERSE_SCALE);
            
            bool occupied = false;
            float disterr = 10e-6f;
            for (int j = 0; j < i; j++) {
                occupied = (abs(m_position[3 * j] - m_position[3 * i]) < disterr) &&
                    (abs(m_position[3 * j + 1] - m_position[3 * i + 1]) < disterr) &&
                    (abs(m_position[3 * j + 2] - m_position[3 * i + 2]) < disterr);
                if (occupied) break;
            }
            if (occupied) continue;

            m_mass[i] = G * (float)((rand() % MASS_SCALE) + 1.0f);
            
            sign = (rand() % 2) ? -1.0 : 1.0;
            m_velocity[3 * i] = (float)((rand() % (VELOCITY_SCALE)) / VELOCITY_SCALE);
            sign = (rand() % 2) ? -1.0 : 1.0;
            m_velocity[3 * i + 1] = (float)((rand() % (VELOCITY_SCALE)) / VELOCITY_SCALE);
            sign = (rand() % 2) ? -1.0 : 1.0;
            m_velocity[3 * i + 2] = (float)((rand() % (VELOCITY_SCALE)) / VELOCITY_SCALE);

            m_acceleration[3 * i] = 0.0f;
            m_acceleration[3 * i + 1] = 0.0f;
            m_acceleration[3 * i + 2] = 0.0f;

            i++;
        }
        break;
    default:
        break;
    }
    

    m_systemInitialized = true;
}

void BodySystem::initGL(int *argc, char* argv[]) {
    glutInit(argc, argv);
    glutInitWindowSize(650, 650);
    glutInitWindowPosition(0, 0);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
    glutCreateWindow("N-test szimuláció");
}

void BodySystem::setAlgorithm(AlgorithmType typeIn) {
    if (m_algorithmInitialized) {
        delete m_algorithm;
    }
    switch (typeIn) {
    case(ALL_PAIRS) :
        m_algorithm = new NBodyAlgorithmAllPairs;
        m_algorithmInitialized = true;
        break;

    case(ALL_PAIRS_SELECTIVE) :
        break;

    default:
        break;
    }
}

void BodySystem::integrate(float startTime, float endTime, float stepTime) {
    assert(m_systemInitialized);
    assert(m_algorithmInitialized);
    assert(startTime < endTime);

    for (float i = startTime; i < endTime;) {
        m_algorithm->advance(m_numBody, m_mass, m_position, m_velocity, m_acceleration, stepTime);

        //std::cout << "#############################################################" << std::endl;
        //std::cout << "Time: " << i << std::endl;

        /*for (int j = 0; j < m_numBody; j++) {
            std::cout << j << " Pos: (" << m_position[j] << ", " << m_position[j + 1] << ", " << m_position[j + 2] << ")" << std::endl;
        }*/

        renderSystem(m_numBody, m_position);

        if ((i + stepTime) > endTime) {
            stepTime = endTime - i;
            i = endTime;
        }
        else {
            i += stepTime;
        }
    }
}
void BodySystem::renderSystem(const unsigned int numBody, const float *pos) {
    /* Ablak torlese az aktualis torloszinnel. */
    glClear(GL_COLOR_BUFFER_BIT);

    /* Poligon megjelenitesi modja. */
    int filled_display = 1;
    if (filled_display)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    else
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

    /*GLfloat sizes[2] = {0.1f, 2.0f};
    const GLfloat quadratic[] = { 0.0f, 0.0f, 0.01f };

    glGetFloatv(GL_ALIASED_POINT_SIZE_RANGE, sizes);
    glEnable(GL_POINT_SPRITE_ARB);
    glPointParameterfARB(GL_POINT_SIZE_MAX_ARB, sizes[1]);
    glPointParameterfARB(GL_POINT_SIZE_MIN_ARB, sizes[0]);
    glPointParameterfvARB(GL_POINT_DISTANCE_ATTENUATION_ARB, quadratic);
    glTexEnvi(GL_POINT_SPRITE_ARB, GL_COORD_REPLACE_ARB, GL_TRUE);*/
    glPointSize(2.f);
    glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
    glBegin(GL_POINTS);
    for (int i = 0; i < numBody; i++)
    {
        glVertex3f(pos[3 * i] / UNIVERSE_SCALE/10, pos[3 * i + 1] / UNIVERSE_SCALE/10, pos[3 * i + 2] / UNIVERSE_SCALE/10);
    }
    // Done drawing points
    glEnd();
    glDisable(GL_POINT_SPRITE_ARB);
    /* Pufferek csereje, uj kep megjelenitese */
    glutSwapBuffers();
}

BodySystem::~BodySystem() {
    delete[] m_mass;
    delete[] m_position;
    delete[] m_velocity;
    delete[] m_acceleration;

    delete m_algorithm;
}