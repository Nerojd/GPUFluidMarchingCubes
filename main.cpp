// #include <GL/glew.h>
// #include <cuda_gl_interop.h>

#include <GL/freeglut.h>
// #include "freeglut_std.h"
// #include "../../../../../../../msys64/mingw64/include/GL/glu.h"
// #include "../../../../../../../msys64/mingw64/include/GL/gl.h"

#include <iostream>
#include <stdlib.h>
#include <vector>

#define _USE_MATH_DEFINES
#include <cmath>

#include <gl/GL.h>
#include <math.h>
#include <algorithm>
#include "vector_types.h"

#include <cuda_runtime.h>
// #include <device_launch_parameters.h>

// #include <ParticleSystem.cuh>

const int K_WINDOW_WIDTH = 1200;
const int K_WINDOW_HEIGHT = 800;

const float K_BOX_X = 4.0f;
const float K_BOX_Y = 4.0f;
const float K_BOX_Z = 4.0f;

const int K_NUM_PARTICLES = 1024;
const float K_PARTICLE_RADIUS = 0.2f;
const float K_PARTICLE_SPACING = 0.3f; // Distance between particles
const float K_PARTICLE_MASS = 1.0f;
const float K_BOUNCE_DAMPING = 0.002f;
const float K_GLOBAL_DAMPING = 0.98f;
const float K_REST_DENSITY = 8.0f;
const float K_GAS_CONSTANT = 60.0f; // Pressure rigidity
const float K_VISCOSITY = 0.3f;     // Viscosity coefficient
const float K_H = 0.994f;           // Influence radius (kernel)
const float K_H2 = K_H * K_H;
const float K_XSPH_FACTOR = 0.98f;

const float K_GRAVITY = 9.8f;

// GLuint vbo;
// struct cudaGraphicsResource *cudaVboResource;

// Créer un VBO et l'enregistrer auprès de CUDA
// void createVBO(GLuint *vbo)
// {
//     glGenBuffers(1, vbo);
//     glBindBuffer(GL_ARRAY_BUFFER, *vbo);
//     glBufferData(GL_ARRAY_BUFFER, K_NUM_PARTICLES * sizeof(float3), nullptr, GL_DYNAMIC_DRAW);
//     glBindBuffer(GL_ARRAY_BUFFER, 0);

//     cudaGraphicsGLRegisterBuffer(&cudaVboResource, *vbo, cudaGraphicsMapFlagsWriteDiscard);
// }

class SolidSphere
{
protected:
    std::vector<GLfloat> vertices;
    std::vector<GLfloat> normals;
    std::vector<GLfloat> texcoords;
    std::vector<GLushort> indices;

public:
    SolidSphere() = default;

    SolidSphere(float radius, unsigned int rings, unsigned int sectors)
    {
        float const R = 1. / (float)(rings - 1);
        float const S = 1. / (float)(sectors - 1);

        vertices.clear();
        normals.clear();
        texcoords.clear();
        indices.clear();

        for (unsigned int r = 0; r < rings; r++)
            for (unsigned int s = 0; s < sectors; s++)
            {
                float const y = sin(-M_PI_2 + M_PI * r * R);
                float const x = cos(2 * M_PI * s * S) * sin(M_PI * r * R);
                float const z = sin(2 * M_PI * s * S) * sin(M_PI * r * R);

                texcoords.push_back(s * S);
                texcoords.push_back(r * R);

                vertices.push_back(x * radius);
                vertices.push_back(y * radius);
                vertices.push_back(z * radius);

                normals.push_back(x);
                normals.push_back(y);
                normals.push_back(z);
            }

        for (unsigned int r = 0; r < rings; r++)
            for (unsigned int s = 0; s < sectors; s++)
            {
                indices.push_back(r * sectors + s);
                indices.push_back(r * sectors + (s + 1));
                indices.push_back((r + 1) * sectors + (s + 1));
                indices.push_back((r + 1) * sectors + s);
            }
    }

    void draw(GLfloat x, GLfloat y, GLfloat z)
    {
        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();

        glTranslatef(x, y, z);

        glEnableClientState(GL_VERTEX_ARRAY);
        glEnableClientState(GL_NORMAL_ARRAY);
        // glEnableClientState(GL_TEXTURE_COORD_ARRAY);

        glVertexPointer(3, GL_FLOAT, 0, vertices.data());
        glNormalPointer(GL_FLOAT, 0, normals.data());
        // glTexCoordPointer(2, GL_FLOAT, 0, texcoords.data());

        glDrawElements(GL_QUADS, indices.size(), GL_UNSIGNED_SHORT, indices.data());

        glDisableClientState(GL_VERTEX_ARRAY);
        glDisableClientState(GL_NORMAL_ARRAY);
        // glDisableClientState(GL_TEXTURE_COORD_ARRAY);

        glPopMatrix();
    }
};

struct Vector3
{
    float x, y, z;

    Vector3 operator+(const Vector3 &other) const
    {
        return {x + other.x, y + other.y, z + other.z};
    }

    Vector3 operator*(float scalar) const
    {
        return {x * scalar, y * scalar, z * scalar};
    }

    Vector3 operator=(float3 vector) const
    {
        return {vector.x, vector.y, vector.z};
    }
};

struct Particle
{
    Vector3 position;
    Vector3 velocity;
    Vector3 force;
    float density;
    float pressure;
    SolidSphere sphere;
};

Particle particles[K_NUM_PARTICLES];

float3 *d_pos;
float3 *d_vel;
float3 *d_force;
float *d_density;
float *d_pressure;

bool initParticles()
{
    float sizeX = K_BOX_X - (-K_BOX_X);
    float sizeY = K_BOX_Y - (-K_BOX_Y);
    float sizeZ = K_BOX_Z - (-K_BOX_Z);

    // Nombre de particules possibles par direction
    int maxX = (int)(sizeX / K_PARTICLE_SPACING);
    int maxY = (int)(sizeY / K_PARTICLE_SPACING);
    int maxZ = (int)(sizeZ / K_PARTICLE_SPACING);

    int count = 0;
    int ix = 0, iy = 0, iz = 0;

    while (count < K_NUM_PARTICLES)
    {
        // Calcul des coordonnées en partant du haut
        float x = (-K_BOX_X) + K_PARTICLE_SPACING * ix + K_PARTICLE_SPACING / 2.0f;
        float y = K_BOX_Y - K_PARTICLE_SPACING * iy - K_PARTICLE_SPACING / 2.0f;
        float z = (-K_BOX_Z) + K_PARTICLE_SPACING * iz + K_PARTICLE_SPACING / 2.0f;

        particles[count].sphere = SolidSphere(K_PARTICLE_RADIUS, 12, 24);

        particles[count].position.x = x;
        particles[count].position.y = y;
        particles[count].position.z = z;

        particles[count].velocity.x = 0.0f;
        particles[count].velocity.y = 0.0f;
        particles[count].velocity.z = 0.0f;

        count++;

        // Avancer dans la grille comme un compteur 3D
        iz++;
        if (iz >= maxZ)
        {
            iz = 0;
            ix++;
            if (ix >= maxX)
            {
                ix = 0;
                iy++;
                if (iy >= maxY)
                {
                    // On a dépassé la capacité de la grille
                    break;
                }
            }
        }
    }

    return true;
}

void drawParticles()
{
    for (int i = 0; i < K_NUM_PARTICLES; ++i)
    {
        float t = (particles[i].density - K_REST_DENSITY) / K_REST_DENSITY;
        t = std::clamp(t, 0.0f, 1.0f);
        glColor3f(t, 0.2f, 1.0f - t); // red = +dense, blue = -dense

        particles[i].sphere.draw(particles[i].position.x,
                                 particles[i].position.y,
                                 particles[i].position.z);
    }
}

void drawBox()
{
    glColor3f(0.3f, 0.3f, 0.3f);
    glBegin(GL_LINE_LOOP);
    glVertex3f(-K_BOX_X, -K_BOX_Y, -K_BOX_Z);
    glVertex3f(K_BOX_X, -K_BOX_Y, -K_BOX_Z);
    glVertex3f(K_BOX_X, K_BOX_Y, -K_BOX_Z);
    glVertex3f(-K_BOX_X, K_BOX_Y, -K_BOX_Z);
    glEnd();

    glBegin(GL_LINE_LOOP);
    glVertex3f(-K_BOX_X, -K_BOX_Y, K_BOX_Z);
    glVertex3f(K_BOX_X, -K_BOX_Y, K_BOX_Z);
    glVertex3f(K_BOX_X, K_BOX_Y, K_BOX_Z);
    glVertex3f(-K_BOX_X, K_BOX_Y, K_BOX_Z);
    glEnd();

    glBegin(GL_LINES);
    glVertex3f(-K_BOX_X, -K_BOX_Y, -K_BOX_Z);
    glVertex3f(-K_BOX_X, -K_BOX_Y, K_BOX_Z);

    glVertex3f(K_BOX_X, -K_BOX_Y, -K_BOX_Z);
    glVertex3f(K_BOX_X, -K_BOX_Y, K_BOX_Z);

    glVertex3f(K_BOX_X, K_BOX_Y, -K_BOX_Z);
    glVertex3f(K_BOX_X, K_BOX_Y, K_BOX_Z);

    glVertex3f(-K_BOX_X, K_BOX_Y, -K_BOX_Z);
    glVertex3f(-K_BOX_X, K_BOX_Y, K_BOX_Z);
    glEnd();
}

void display()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    float const win_aspect = (float)K_WINDOW_WIDTH / (float)K_WINDOW_HEIGHT;
    glViewport(0, 0, K_WINDOW_WIDTH, K_WINDOW_HEIGHT);
    gluPerspective(45, win_aspect, 1, 100);

    gluLookAt(0.0, 2.0, -40.0, // eye
              0.0, 0.5, 0.0,   // center
              0.0, 1.0, 0.0);  // height

#ifdef DRAW_WIREFRAME
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
#endif

    drawBox();
    drawParticles();

    glutSwapBuffers();
}

void ResolveWallCollisions(int i)
{
    if (particles[i].position.x < -K_BOX_X)
    {
        particles[i].position.x = -K_BOX_X;
        particles[i].velocity.x *= -K_BOUNCE_DAMPING;
    }
    else if (particles[i].position.x > K_BOX_X)
    {
        particles[i].position.x = K_BOX_X;
        particles[i].velocity.x *= -K_BOUNCE_DAMPING;
    }

    if (particles[i].position.y < -K_BOX_Y)
    {
        particles[i].position.y = -K_BOX_Y;
        particles[i].velocity.y *= -K_BOUNCE_DAMPING;
    }
    else if (particles[i].position.y > K_BOX_Y)
    {
        particles[i].position.y = K_BOX_Y;
        particles[i].velocity.y *= -K_BOUNCE_DAMPING;
    }

    if (particles[i].position.z < -K_BOX_Z)
    {
        particles[i].position.z = -K_BOX_Z;
        particles[i].velocity.z *= -K_BOUNCE_DAMPING;
    }
    else if (particles[i].position.z > K_BOX_Z)
    {
        particles[i].position.z = K_BOX_Z;
        particles[i].velocity.z *= -K_BOUNCE_DAMPING;
    }
}

float poly6(float r2)
{
    float hr2 = K_H2 - r2;
    return (315.0f / (64.0f * M_PI * pow(K_H, 9))) * pow(hr2, 3);
}

float spikyGradient(float r)
{
    float hr = K_H - r;
    return (-45.0f / (M_PI * pow(K_H, 6))) * hr * hr;
}

float viscosityLaplacian(float r)
{
    return (45.0f / (M_PI * pow(K_H, 6))) * (K_H - r);
}

void computeDensityPressure()
{
    for (int i = 0; i < K_NUM_PARTICLES; ++i)
    {
        Particle &particleI = particles[i];
        particleI.density = 0.0f;

        for (int j = 0; j < K_NUM_PARTICLES; ++j)
        {
            Particle &particleJ = particles[j];
            float dx = particleI.position.x - particleJ.position.x;
            float dy = particleI.position.y - particleJ.position.y;
            float dz = particleI.position.z - particleJ.position.z;
            float r2 = dx * dx + dy * dy + dz * dz;

            if (r2 < K_H2)
            {
                particleI.density += K_PARTICLE_MASS * poly6(r2);
            }
        }

        particleI.pressure = K_GAS_CONSTANT * (particleI.density - K_REST_DENSITY);
    }
}

void computeForces()
{
    for (int i = 0; i < K_NUM_PARTICLES; ++i)
    {
        Particle &particleI = particles[i];
        particleI.force.x = particleI.force.y = particleI.force.z = 0.0f;

        for (int j = 0; j < K_NUM_PARTICLES; ++j)
        {
            if (i == j)
                continue;

            Particle &particleJ = particles[j];
            float dx = particleI.position.x - particleJ.position.x;
            float dy = particleI.position.y - particleJ.position.y;
            float dz = particleI.position.z - particleJ.position.z;

            float r2 = dx * dx + dy * dy + dz * dz;
            if (r2 < K_H2 && r2 > 0.0001f)
            {
                float r = sqrt(r2);
                float nx = dx / r, ny = dy / r, nz = dz / r;

                // Pression
                float pterm = -K_PARTICLE_MASS * (particleI.pressure + particleJ.pressure) / (2.0f * particleJ.density);
                float grad = spikyGradient(r);
                particleI.force.x += pterm * grad * nx;
                particleI.force.y += pterm * grad * ny;
                particleI.force.z += pterm * grad * nz;

                // Viscosité
                float vx = particleJ.velocity.x - particleI.velocity.x;
                float vy = particleJ.velocity.y - particleI.velocity.y;
                float vz = particleJ.velocity.z - particleI.velocity.z;
                float lap = viscosityLaplacian(r);
                float vterm = K_VISCOSITY * K_PARTICLE_MASS / particleJ.density;

                particleI.force.x += vterm * lap * vx;
                particleI.force.y += vterm * lap * vy;
                particleI.force.z += vterm * lap * vz;
            }
        }

        // Ajouter gravité
        particleI.force.y += -9.8f * particleI.density;
    }
}

void integrate(float dt)
{
    for (int i = 0; i < K_NUM_PARTICLES; ++i)
    {
        Particle &p = particles[i];
        p.velocity.x += dt * p.force.x / p.density;
        p.velocity.y += dt * p.force.y / p.density;
        p.velocity.z += dt * p.force.z / p.density;

        p.position.x += dt * p.velocity.x;
        p.position.y += dt * p.velocity.y;
        p.position.z += dt * p.velocity.z;

        // Walls collision
        ResolveWallCollisions(i);
    }
}

void XsphViscosity()
{
    for (int i = 0; i < K_NUM_PARTICLES; ++i)
    {
        Particle &particleI = particles[i];

        for (int j = 0; j < K_NUM_PARTICLES; ++j)
        {
            if (i == j)
                continue;

            Particle &particleJ = particles[j];
            float dx = particleI.position.x - particleJ.position.x;
            float dy = particleI.position.y - particleJ.position.y;
            float dz = particleI.position.z - particleJ.position.z;
            float r2 = dx * dx + dy * dy + dz * dz;

            if (r2 < K_H2 && r2 > 0.0001f)
            {
                float r = sqrt(r2);
                float w = poly6(r2);

                // XSPH correction
                particleI.velocity.x += K_XSPH_FACTOR * w * (particleJ.velocity.x - particleI.velocity.x);
                particleI.velocity.y += K_XSPH_FACTOR * w * (particleJ.velocity.y - particleI.velocity.y);
                particleI.velocity.z += K_XSPH_FACTOR * w * (particleJ.velocity.z - particleI.velocity.z);
            }
        }

        // particleI.velocity.x *= K_GLOBAL_DAMPING;
        // particleI.velocity.y *= K_GLOBAL_DAMPING;
        // particleI.velocity.z *= K_GLOBAL_DAMPING;
    }
}

void idle()
{
    static int lastTime = glutGet(GLUT_ELAPSED_TIME);
    int time = glutGet(GLUT_ELAPSED_TIME);
    float deltaTime = (time - lastTime) / 1200.0f;
    lastTime = time;

    computeDensityPressure();
    computeForces();
    integrate(deltaTime);
    XsphViscosity();

    glutPostRedisplay();
}

bool initGL(int argc, char **argv)
{
    std::cout << "glutInit" << std::endl;
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowSize(K_WINDOW_WIDTH, K_WINDOW_HEIGHT);
    glutInitWindowPosition(200, 200);

    std::cout << "glutCreateWindow" << std::endl;
    int windowId = glutCreateWindow("CUDA + OpenGL Particles");

    if (windowId <= 0)
    {
        std::cerr << "Failed to create GLUT window" << std::endl;
        return false;
    }

    glClearColor(0.0, 0.0, 0.0, 1.0); // fond noir
    glPointSize(2.0);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    std::cout << "glutMatrix" << std::endl;
    // glMatrixMode(GL_PROJECTION);
    // glLoadIdentity();
    // gluOrtho2D(-1, 1, -1, 1); // coordonnées normalisées

    std::cout << "end of initGL" << std::endl;

    return true;
}

bool initCUDA()
{
    cudaError_t err = cudaMalloc(&d_pos, K_NUM_PARTICLES * sizeof(float3));
    if (err != cudaSuccess)
    {
        std::cerr << "Failed to allocate device memory for d_pos" << std::endl;
        return false;
    }
    err = cudaMalloc(&d_vel, K_NUM_PARTICLES * sizeof(float3));
    if (err != cudaSuccess)
    {
        std::cerr << "Failed to allocate device memory for d_vel" << std::endl;
        return false;
    }
    err = cudaMalloc(&d_force, K_NUM_PARTICLES * sizeof(float3));
    if (err != cudaSuccess)
    {
        std::cerr << "Failed to allocate device memory for d_force" << std::endl;
        return false;
    }
    err = cudaMalloc(&d_density, K_NUM_PARTICLES * sizeof(float));
    if (err != cudaSuccess)
    {
        std::cerr << "Failed to allocate device memory for d_density" << std::endl;
        return false;
    }
    err = cudaMalloc(&d_pressure, K_NUM_PARTICLES * sizeof(float));
    if (err != cudaSuccess)
    {
        std::cerr << "Failed to allocate device memory for d_pressure" << std::endl;
        return false;
    }

    return true;
}

void keyboard(unsigned char key, int x, int y)
{
    if (key == 27)
        exit(0); // ESC KEY
}

int main(int argc, char **argv)
{
    std::cout << "Hello World" << std::endl;

    bool status;

    // initialize OpenGL Windows
    status = initGL(argc, argv);

    // initialize Particles on CUDA
    status &= initParticles();

    // initialize CUDA
    status &= initCUDA();

    // createVBO(&vbo);

    if (status)
    {
        std::cout << "Start Update" << std::endl;
        // OpenGL Renderer
        glutDisplayFunc(display);
        glutIdleFunc(idle);
        glutKeyboardFunc(keyboard);
        glutMainLoop();

        // std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    // Clean all
    std::cout << "Cleanup the project" << std::endl;
    // cleanupParticles();
    // cleanupRenderer();
    return 0;
}

// CTRL SHIFT B
// F5