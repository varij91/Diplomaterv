#include "NBodyRenderer.h"

#define GLEW_STATIC
#include <GL\glew.h>
#include <GL\freeglut.h>

std::shared_ptr<NBodyProperties> NBodyRenderer::mp_properties;
std::shared_ptr<NBodySystem> NBodyRenderer::mp_system;

void NBodyRenderer::renderCallback() {
    renderSystem();

    mp_system->advance();

    if (mp_properties->currentTime >= mp_properties->endTime) {
        glutLeaveMainLoop();
    }
}

void NBodyRenderer::idleCallback() {
    glutPostRedisplay();
}

void NBodyRenderer::renderMainLoop() {
    glutDisplayFunc(renderCallback);
    glutIdleFunc(idleCallback);
    glutMainLoop();
}

void NBodyRenderer::initGL(int *argc, char* argv[]) {
    glutInit(argc, argv);
    // Törlési szín beállítása
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glutInitWindowSize(720, 720);
    glutInitWindowPosition(0, 0);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    // Pontok simításának beállítása
    glEnable(GL_POINT_SMOOTH);
    glHint(GL_POINT_SMOOTH_HINT, GL_FASTEST);

    glClearDepth(1.0f);        // Set background depth to farthest
    glEnable(GL_DEPTH_TEST);   // Enable depth testing for z-culling
    glDepthFunc(GL_LEQUAL);    // Set the type of depth-test
    glShadeModel(GL_SMOOTH);   // Enable smooth shading
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);  // Nice perspective corrections

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    //gluPerspective(45.0f, aspect, 0.1f, 100.0f);
    /*glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-10.0f, 10.0f, -10.0f, 10.0f, -10.0f, 10.0f);*/
    //glOrtho(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);

    glutCreateWindow("N-test szimuláció");
}

void NBodyRenderer::renderSystem(){
    //glClear(GL_COLOR_BUFFER_BIT);
    //glEnable(GL_TEXTURE_2D);
    //glBindTexture(GL_TEXTURE_2D, 1);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    //float size = mp_properties->positionScale/500.0f;

    ///*glEnable(GL_BLEND);
    //glEnable(GL_POLYGON_SMOOTH);
    //glBlendFunc(GL_SRC_ALPHA_SATURATE,GL_ZERO);
    //glHint(GL_POLYGON_SMOOTH_HINT,GL_FASTEST);
    //glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    //glEnable(GL_MULTISAMPLE_ARB);*/

    //glMatrixMode(GL_PROJECTION);
    //glLoadIdentity();
    //glFrustum(-0.1, 0.1, -0.1, 0.1, 0.1, 100.0);
    //glMatrixMode(GL_MODELVIEW);
    //glLoadIdentity();

    //glEnable(GL_ALPHA_TEST);
    //glEnable(GL_POLYGON_SMOOTH);
    //glEnable(GL_POINT_SMOOTH);
    //glEnable(GL_BLEND);
    //glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    ////glPointSize(4.0f);
    ///*glBegin(GL_POINTS);

    //for (int i = 0; i < mp_properties->numBody; i++)
    //{
    //    glVertex3f(m_bodies.at(i).position.x, m_bodies.at(i).position.y, m_bodies.at(i).position.z);
    //}
    //glEnd();*/
    //glBegin(GL_QUADS);
    //for (int i = 0; i < mp_properties->numBody; i++)
    //{

    //    glTexCoord2f(0, 0);
    //    glVertex3f(m_bodies.at(i).position.x - size, m_bodies.at(i).position.y - size, m_bodies.at(i).position.z);
    //    glTexCoord2f(0, 1);
    //    glVertex3f(m_bodies.at(i).position.x - size, m_bodies.at(i).position.y + size, m_bodies.at(i).position.z);
    //    glTexCoord2f(1, 1);
    //    glVertex3f(m_bodies.at(i).position.x + size, m_bodies.at(i).position.y + size, m_bodies.at(i).position.z);
    //    glTexCoord2f(1, 0);
    //    glVertex3f(m_bodies.at(i).position.x + size, m_bodies.at(i).position.y - size, m_bodies.at(i).position.z);
    //    
    //}
    //glEnd();
    //glutSwapBuffers();

    glClear(GL_COLOR_BUFFER_BIT);
    glEnable(GL_POINT_SMOOTH);
    //glPointSize(9.0f);
    /*glEnable(GL_FOG);
    glFogf(GL_FOG_MODE, GL_EXP);
    glFogf(GL_FOG_COLOR, (1.0, 1.0, 1.0, 0.0));*/
    glBegin(GL_POINTS);

    for (int i = 0; i < mp_properties->numBody; i++)
    {
        unsigned int red, green, blue;
        getColor(red, green, blue, i);
        glColor3ui(red << 24, green << 24, blue << 24);
        glVertex3f(mp_system->m_bodies.at(i).position.x / (mp_properties->positionScale * 10), mp_system->m_bodies.at(i).position.y / (mp_properties->positionScale * 10), mp_system->m_bodies.at(i).position.z / (mp_properties->positionScale * 10)); 
    }
    glEnd();

    glutSwapBuffers();
}

void NBodyRenderer::getColor(unsigned int &r, unsigned int &g, unsigned int &b, int index) {
    /*unsigned int value = (mp_properties->numNeighbours.at(index) * 256) / mp_properties->numBody;
    unsigned int quotient = value / 256;
    unsigned int remainder = value % 256;
    if (quotient == 0) {
    r = remainder; g = 0; b = 255;
    }
    else if (quotient == 1) {
    r = 255; g = 0; b = 255-remainder;
    }
    else if(quotient == 2) {
    r = 255; g = remainder; b = 0;
    }
    else {
    r = 255; g = 255; b = 0;
    }*/

    float value = (float)mp_properties->numNeighbours.at(index) / (float)mp_properties->numBody;
    if (mp_properties->numBody <= 128)
        value = value * 0.25f;
    else if (mp_properties->numBody <= 256)
        value = value * 0.5f;
    else if (mp_properties->numBody <= 512)
        value = value * 1.0f;
    else if (mp_properties->numBody <= 1024)
        value = value * 2.2f;
    else if (mp_properties->numBody <= 2048)
        value = value * 2.5f;
    else if (mp_properties->numBody <= 4096)
        value = value * 5.0f;
    else
        value = value * 16.0f;

    if (value < 0.001f) {
        r = 0; g = 0; b = 255;
    }
    else if (value < 0.002f) {
        r = 127; g = 0; b = 255;
    }
    else if (value < 0.004f) {
        r = 255; g = 0; b = 255;
    }
    else if (value < 0.008f) {
        r = 255; g = 0; b = 170;
    }
    else if (value < 0.016f) {
        r = 255; g = 0; b = 85;
    }
    else if (value < 0.032f) {
        r = 255; g = 0; b = 0;
    }
    else if (value < 0.064f) {
        r = 255; g = 31; b = 0;
    }
    else if (value < 0.128f) {
        r = 255; g = 63; b = 0;
    }
    else if (value < 0.256f) {
        r = 255; g = 95; b = 0;
    }
    else if (value < 0.384f) {
        r = 255; g = 127; b = 0;
    }
    else if (value < 0.512f) {
        r = 255; g = 159; b = 0;
    }
    else if (value < 0.640f) {
        r = 255; g = 191; b = 0;
    }
    else if (value < 0.768f) {
        r = 255; g = 223; b = 0;
    }
    else {
        r = 255; g = 255; b = 0;
    }
}
