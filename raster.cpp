#include <iostream>
#define _USE_MATH_DEFINES
#include <cmath>

#include "GamesEngineeringBase.h" // Include the GamesEngineeringBase header
#include <algorithm>
#include <chrono>
#include <vector>
#include<thread>
#include <cmath>
#include "matrix.h"
#include "colour.h"
#include "mesh.h"
#include "zbuffer.h"
#include "renderer.h"
#include "RNG.h"
#include "light.h"
#include "triangle.h"
#include"MultiThreading.h"
// Main rendering function that processes a mesh, transforms its vertices, applies lighting, and draws triangles on the canvas.
// Input Variables:
// - renderer: The Renderer object used for drawing.
// - mesh: Pointer to the Mesh object containing vertices and triangles to render.
// - camera: Matrix representing the camera's transformation.
// - L: Light object representing the lighting parameters.
void render(Renderer& renderer, Mesh* mesh, matrix& camera, Light& L) {
    // Combine perspective, camera, and world transformations for the mesh
    matrix p = renderer.perspective * camera * mesh->world;

    // Iterate through all triangles in the mesh
    for (triIndices& ind : mesh->triangles) {
        Vertex t[3]; // Temporary array to store transformed triangle vertices

        // Transform each vertex of the triangle
        for (unsigned int i = 0; i < 3; i++) {
            t[i].p = p * mesh->vertices[ind.v[i]].p; // Apply transformations
            t[i].p.divideW(); // Perspective division to normalize coordinates

            // Transform normals into world space for accurate lighting
            // no need for perspective correction as no shearing or non-uniform scaling
            t[i].normal = mesh->world * mesh->vertices[ind.v[i]].normal;
            t[i].normal.normalise();

            // Map normalized device coordinates to screen space
            t[i].p[0] = (t[i].p[0] + 1.f) * 0.5f * static_cast<float>(renderer.canvas.getWidth());
            t[i].p[1] = (t[i].p[1] + 1.f) * 0.5f * static_cast<float>(renderer.canvas.getHeight());
            t[i].p[1] = renderer.canvas.getHeight() - t[i].p[1]; // Invert y-axis

            // Copy vertex colours
            t[i].rgb = mesh->vertices[ind.v[i]].rgb;
        }

        // Clip triangles with Z-values outside [-1, 1]
        if (fabs(t[0].p[2]) > 1.0f || fabs(t[1].p[2]) > 1.0f || fabs(t[2].p[2]) > 1.0f) continue;

        // Create a triangle object and render it
        triangle tri(t[0], t[1], t[2]);
        tri.draw(renderer, L, mesh->ka, mesh->kd);
    }
}

// using Mesh& instead of Mesh*
void render1(Renderer& renderer, Mesh& mesh, matrix& camera, Light& L) {
    // Combine perspective, camera, and world transformations for the mesh
    matrix p = renderer.perspective * camera * mesh.world;

    // Iterate through all triangles in the mesh
    for (triIndices& ind : mesh.triangles) {
        Vertex t[3]; // Temporary array to store transformed triangle vertices

        // Transform each vertex of the triangle
        for (unsigned int i = 0; i < 3; i++) {
            t[i].p = p * mesh.vertices[ind.v[i]].p; // Apply transformations
            t[i].p.divideW(); // Perspective division to normalize coordinates

            // Transform normals into world space for accurate lighting
            // no need for perspective correction as no shearing or non-uniform scaling
            t[i].normal = mesh.world * mesh.vertices[ind.v[i]].normal;
            t[i].normal.normalise();

            // Map normalized device coordinates to screen space
            t[i].p[0] = (t[i].p[0] + 1.f) * 0.5f * static_cast<float>(renderer.canvas.getWidth());
            t[i].p[1] = (t[i].p[1] + 1.f) * 0.5f * static_cast<float>(renderer.canvas.getHeight());
            t[i].p[1] = renderer.canvas.getHeight() - t[i].p[1]; // Invert y-axis

            // Copy vertex colours
            t[i].rgb = mesh.vertices[ind.v[i]].rgb;
        }

        // Clip triangles with Z-values outside [-1, 1]
        if (fabs(t[0].p[2]) > 1.0f || fabs(t[1].p[2]) > 1.0f || fabs(t[2].p[2]) > 1.0f) continue;

        // Create a triangle object and render it
        triangle tri(t[0], t[1], t[2]);
        tri.draw(renderer, L, mesh.ka, mesh.kd);
    }
}

//Using Mesh& instead of Mesh*
//Memorization: Pre-calculating the transformation of vertices
void render2(Renderer& renderer, Mesh& mesh, matrix& camera, Light& L) {
    // Combine perspective, camera, and world transformations for the mesh
    matrix p = renderer.perspective * camera * mesh.world;

    // Transform each vertex of the mesh
    int size = mesh.vertices.size();
    std::vector<Vertex> transformedVertices(size);

    for (int i = 0; i < size; i++) {
        transformedVertices[i].p = p * mesh.vertices[i].p; // Apply transformations
        transformedVertices[i].p.divideW(); // Perspective division to normalize coordinates

        // Transform normals into world space for accurate lighting
        // no need for perspective correction as no shearing or non-uniform scaling
        transformedVertices[i].normal = mesh.world * mesh.vertices[i].normal;
        transformedVertices[i].normal.normalise();

        // Map normalized device coordinates to screen space
        transformedVertices[i].p[0] = (transformedVertices[i].p[0] + 1.f) * 0.5f * static_cast<float>(renderer.canvas.getWidth());
        transformedVertices[i].p[1] = (transformedVertices[i].p[1] + 1.f) * 0.5f * static_cast<float>(renderer.canvas.getHeight());
        transformedVertices[i].p[1] = renderer.canvas.getHeight() - transformedVertices[i].p[1]; // Invert y-axis

        // Copy vertex colours
        transformedVertices[i].rgb = mesh.vertices[i].rgb;
    }

    // Iterate through all triangles in the mesh
    for (triIndices& ind : mesh.triangles) {
        // Clip triangles with Z-values outside [-1, 1]
        if (fabs(transformedVertices[ind.v[0]].p[2]) > 1.0f || fabs(transformedVertices[ind.v[1]].p[2]) > 1.0f || fabs(transformedVertices[ind.v[2]].p[2]) > 1.0f) continue;

        // Create a triangle object and render it
        triangle tri(transformedVertices[ind.v[0]], transformedVertices[ind.v[1]], transformedVertices[ind.v[2]]);
        tri.draw(renderer, L, mesh.ka, mesh.kd);
    }
}

//using draw1 for optimizating algorithm for rasterization
void render3(Renderer& renderer, Mesh& mesh, matrix& camera, Light& L) {
    // Combine perspective, camera, and world transformations for the mesh
    matrix p = renderer.perspective * camera * mesh.world;

    // Iterate through all triangles in the mesh
    for (triIndices& ind : mesh.triangles) {
        Vertex t[3]; // Temporary array to store transformed triangle vertices

        // Transform each vertex of the triangle
        for (unsigned int i = 0; i < 3; i++) {
            t[i].p = p * mesh.vertices[ind.v[i]].p; // Apply transformations
            t[i].p.divideW(); // Perspective division to normalize coordinates

            // Transform normals into world space for accurate lighting
            // no need for perspective correction as no shearing or non-uniform scaling
            t[i].normal = mesh.world * mesh.vertices[ind.v[i]].normal;
            t[i].normal.normalise();

            // Map normalized device coordinates to screen space
            t[i].p[0] = (t[i].p[0] + 1.f) * 0.5f * static_cast<float>(renderer.canvas.getWidth());
            t[i].p[1] = (t[i].p[1] + 1.f) * 0.5f * static_cast<float>(renderer.canvas.getHeight());
            t[i].p[1] = renderer.canvas.getHeight() - t[i].p[1]; // Invert y-axis

            // Copy vertex colours
            t[i].rgb = mesh.vertices[ind.v[i]].rgb;
        }

        // Clip triangles with Z-values outside [-1, 1]
        if (fabs(t[0].p[2]) > 1.0f || fabs(t[1].p[2]) > 1.0f || fabs(t[2].p[2]) > 1.0f) continue;

        // Create a triangle object and render it
        triangle tri(t[0], t[1], t[2]);
        tri.draw1(renderer, L, mesh.ka, mesh.kd);
    }
}
//Using draw1 (optimizating algorithm for rasterization)
//Adding backface culling
void render4(Renderer& renderer, Mesh& mesh, matrix& camera, Light& L) {
    // Combine perspective, camera, and world transformations for the mesh
    matrix p = renderer.perspective * camera * mesh.world;

    // Iterate through all triangles in the mesh
    for (triIndices& ind : mesh.triangles) {
        Vertex t[3]; // Temporary array to store transformed triangle vertices

        if (vec4::dot(mesh.world * mesh.vertices[ind.v[0]].normal, mesh.world * mesh.vertices[ind.v[0]].p - vec4(0.0f, 0.0f, -camera.getTranslationZ(), 1.0f)) >= 0.0f) continue;

        // Transform each vertex of the triangle
        for (unsigned int i = 0; i < 3; i++) {
            t[i].p = p * mesh.vertices[ind.v[i]].p; // Apply transformations
            t[i].p.divideW(); // Perspective division to normalize coordinates

            // Transform normals into world space for accurate lighting
            // no need for perspective correction as no shearing or non-uniform scaling
            t[i].normal = mesh.world * mesh.vertices[ind.v[i]].normal;
            t[i].normal.normalise();

            // Map normalized device coordinates to screen space
            t[i].p[0] = (t[i].p[0] + 1.f) * 0.5f * static_cast<float>(renderer.canvas.getWidth());
            t[i].p[1] = (t[i].p[1] + 1.f) * 0.5f * static_cast<float>(renderer.canvas.getHeight());
            t[i].p[1] = renderer.canvas.getHeight() - t[i].p[1]; // Invert y-axis

            // Copy vertex colours
            t[i].rgb = mesh.vertices[ind.v[i]].rgb;
        }

        // Clip triangles with Z-values outside [-1, 1]
        if (fabs(t[0].p[2]) > 1.0f || fabs(t[1].p[2]) > 1.0f || fabs(t[2].p[2]) > 1.0f) continue;

        // Create a triangle object and render it
        triangle tri(t[0], t[1], t[2]);
        tri.draw1(renderer, L, mesh.ka, mesh.kd);
    }
}

//Using draw2 (optimizating algorithm for rasterization) + (skip when minV>maxV)
//Adding backface culling
void render5(Renderer& renderer, Mesh& mesh, matrix& camera, Light& L) {
    // Combine perspective, camera, and world transformations for the mesh
    matrix p = renderer.perspective * camera * mesh.world;

    // Iterate through all triangles in the mesh
    for (triIndices& ind : mesh.triangles) {
        Vertex t[3]; // Temporary array to store transformed triangle vertices

        if (vec4::dot(mesh.world * mesh.vertices[ind.v[0]].normal, mesh.world * mesh.vertices[ind.v[0]].p - vec4(0.0f, 0.0f, -camera.getTranslationZ(), 1.0f)) >= 0.0f) continue;

        // Transform each vertex of the triangle
        for (unsigned int i = 0; i < 3; i++) {
            t[i].p = p * mesh.vertices[ind.v[i]].p; // Apply transformations
            t[i].p.divideW(); // Perspective division to normalize coordinates

            // Transform normals into world space for accurate lighting
            // no need for perspective correction as no shearing or non-uniform scaling
            t[i].normal = mesh.world * mesh.vertices[ind.v[i]].normal;
            t[i].normal.normalise();

            // Map normalized device coordinates to screen space
            t[i].p[0] = (t[i].p[0] + 1.f) * 0.5f * static_cast<float>(renderer.canvas.getWidth());
            t[i].p[1] = (t[i].p[1] + 1.f) * 0.5f * static_cast<float>(renderer.canvas.getHeight());
            t[i].p[1] = renderer.canvas.getHeight() - t[i].p[1]; // Invert y-axis

            // Copy vertex colours
            t[i].rgb = mesh.vertices[ind.v[i]].rgb;
        }

        // Clip triangles with Z-values outside [-1, 1]
        if (fabs(t[0].p[2]) > 1.0f || fabs(t[1].p[2]) > 1.0f || fabs(t[2].p[2]) > 1.0f) continue;

        // Create a triangle object and render it
        triangle tri(t[0], t[1], t[2]);
        tri.draw2(renderer, L, mesh.ka, mesh.kd);
    }
}
// Test scene function to demonstrate rendering with user-controlled transformations
// No input variables
void sceneTest() {
    Renderer renderer;
    // create light source {direction, diffuse intensity, ambient intensity}
    Light L{ vec4(0.f, 1.f, 1.f, 0.f), colour(1.0f, 1.0f, 1.0f), colour(0.1f, 0.1f, 0.1f) };
    // camera is just a matrix
    matrix camera = matrix::makeIdentity(); // Initialize the camera with identity matrix

    bool running = true; // Main loop control variable

    std::vector<Mesh*> scene; // Vector to store scene objects

    // Create a sphere and a rectangle mesh
    Mesh mesh = Mesh::makeSphere(1.0f, 10, 20);
    //Mesh mesh2 = Mesh::makeRectangle(-2, -1, 2, 1);

    // add meshes to scene
    scene.push_back(&mesh);
    // scene.push_back(&mesh2); 

    float x = 0.0f, y = 0.0f, z = -4.0f; // Initial translation parameters
    mesh.world = matrix::makeTranslation(x, y, z);
    //mesh2.world = matrix::makeTranslation(x, y, z) * matrix::makeRotateX(0.01f);

    // Main rendering loop
    while (running) {
        renderer.canvas.checkInput(); // Handle user input
        renderer.clear(); // Clear the canvas for the next frame

        // Apply transformations to the meshes
     //   mesh2.world = matrix::makeTranslation(x, y, z) * matrix::makeRotateX(0.01f);
        mesh.world = matrix::makeTranslation(x, y, z);

        // Handle user inputs for transformations
        if (renderer.canvas.keyPressed(VK_ESCAPE)) break;
        if (renderer.canvas.keyPressed('A')) x += -0.1f;
        if (renderer.canvas.keyPressed('D')) x += 0.1f;
        if (renderer.canvas.keyPressed('W')) y += 0.1f;
        if (renderer.canvas.keyPressed('S')) y += -0.1f;
        if (renderer.canvas.keyPressed('Q')) z += 0.1f;
        if (renderer.canvas.keyPressed('E')) z += -0.1f;

        // Render each object in the scene
        for (auto& m : scene)
            render(renderer, m, camera, L);

        renderer.present(); // Display the rendered frame
    }
}

// Utility function to generate a random rotation matrix
// No input variables
matrix makeRandomRotation() {
    RandomNumberGenerator& rng = RandomNumberGenerator::getInstance();
    unsigned int r = rng.getRandomInt(0, 3);

    switch (r) {
    case 0: return matrix::makeRotateX(rng.getRandomFloat(0.f, 2.0f * M_PI));
    case 1: return matrix::makeRotateY(rng.getRandomFloat(0.f, 2.0f * M_PI));
    case 2: return matrix::makeRotateZ(rng.getRandomFloat(0.f, 2.0f * M_PI));
    default: return matrix::makeIdentity();
    }
}

// Function to render a scene with multiple objects and dynamic transformations
// No input variables
void scene1() {
    Renderer renderer;
    matrix camera;
    Light L{ vec4(0.f, 1.f, 1.f, 0.f), colour(1.0f, 1.0f, 1.0f), colour(0.1f, 0.1f, 0.1f) };

    bool running = true;

    std::vector<Mesh*> scene;

    // Create a scene of 40 cubes with random rotations
    for (unsigned int i = 0; i < 20; i++) {
        Mesh* m = new Mesh();
        *m = Mesh::makeCube(1.f);
        m->world = matrix::makeTranslation(-2.0f, 0.0f, (-3 * static_cast<float>(i))) * makeRandomRotation();
        scene.push_back(m);
        m = new Mesh();
        *m = Mesh::makeCube(1.f);
        m->world = matrix::makeTranslation(2.0f, 0.0f, (-3 * static_cast<float>(i))) * makeRandomRotation();
        scene.push_back(m);
    }

    float zoffset = 8.0f; // Initial camera Z-offset
    float step = -0.1f;  // Step size for camera movement

    auto start = std::chrono::high_resolution_clock::now();
    std::chrono::time_point<std::chrono::high_resolution_clock> end;
    int cycle = 0;

    // Main rendering loop
    while (running) {
        renderer.canvas.checkInput();
        renderer.clear();

        camera = matrix::makeTranslation(0, 0, -zoffset); // Update camera position

        // Rotate the first two cubes in the scene
        scene[0]->world = scene[0]->world * matrix::makeRotateXYZ(0.1f, 0.1f, 0.0f);
        scene[1]->world = scene[1]->world * matrix::makeRotateXYZ(0.0f, 0.1f, 0.2f);

        if (renderer.canvas.keyPressed(VK_ESCAPE)) break;

        zoffset += step;
        if (zoffset < -60.f || zoffset > 8.f) {
            step *= -1.f;
            if (++cycle % 2 == 0) {
                end = std::chrono::high_resolution_clock::now();
                std::cout << cycle / 2 << " :" << std::chrono::duration<double, std::milli>(end - start).count() << "ms\n";
                start = std::chrono::high_resolution_clock::now();
            }
        }

        for (auto& m : scene)
            render(renderer, m, camera, L);
        renderer.present();
    }

    for (auto& m : scene)
        delete m;
}

//Using Mesh instead of Mesh*
//Using render1 (Passing Mesh by reference)
void scene1_1() {
    Renderer renderer;
    matrix camera;
    Light L{ vec4(0.f, 1.f, 1.f, 0.f), colour(1.0f, 1.0f, 1.0f), colour(0.1f, 0.1f, 0.1f) };
    bool running = true;

    std::vector<Mesh> scene(40);

    for (unsigned int i = 0; i < 20; i++) {
        scene[i * 2] = Mesh::makeCube(1.f);
        scene[i * 2].world = matrix::makeTranslation(-2.0f, 0.0f, (-3 * static_cast<float>(i))) * makeRandomRotation();
        scene[i * 2 + 1] = Mesh::makeCube(1.f);
        scene[i * 2 + 1].world = matrix::makeTranslation(2.0f, 0.0f, (-3 * static_cast<float>(i))) * makeRandomRotation();
    }

    float zoffset = 8.0f; // Initial camera Z-offset
    float step = -0.1f;  // Step size for camera movement

    auto start = std::chrono::high_resolution_clock::now();
    std::chrono::time_point<std::chrono::high_resolution_clock> end;
    int cycle = 0;

    // Main rendering loop
    while (running) {
        renderer.canvas.checkInput();
        renderer.clear();

        camera = matrix::makeTranslation(0, 0, -zoffset); // Update camera position

        // Rotate the first two cubes in the scene
        scene[0].world = scene[0].world * matrix::makeRotateXYZ(0.1f, 0.1f, 0.0f);
        scene[1].world = scene[1].world * matrix::makeRotateXYZ(0.0f, 0.1f, 0.2f);

        if (renderer.canvas.keyPressed(VK_ESCAPE)) break;

        zoffset += step;
        if (zoffset < -60.f || zoffset > 8.f) {
            step *= -1.f;
            if (++cycle % 2 == 0) {
                end = std::chrono::high_resolution_clock::now();
                std::cout << cycle / 2 << " :" << std::chrono::duration<double, std::milli>(end - start).count() << "ms\n";
                start = std::chrono::high_resolution_clock::now();
            }
        }

        for (auto& m : scene)
            render1(renderer, m, camera, L);
        renderer.present();
    }

}

//Using Mesh instead of Mesh*
//Using render1 (Passing Mesh by reference)
//Using Memorization of camera matrix
void scene1_2() {
    Renderer renderer;
    matrix camera;
    Light L{ vec4(0.f, 1.f, 1.f, 0.f), colour(1.0f, 1.0f, 1.0f), colour(0.1f, 0.1f, 0.1f) };
    bool running = true;

    std::vector<Mesh> scene(40);

    for (unsigned int i = 0; i < 20; i++) {
        scene[i * 2] = Mesh::makeCube(1.f);
        scene[i * 2].world = matrix::makeTranslation(-2.0f, 0.0f, (-3 * static_cast<float>(i))) * makeRandomRotation();
        scene[i * 2 + 1] = Mesh::makeCube(1.f);
        scene[i * 2 + 1].world = matrix::makeTranslation(2.0f, 0.0f, (-3 * static_cast<float>(i))) * makeRandomRotation();
    }
    //pre-calculate the camera matrix
    std::vector<float> cameraMatrix;
    for (float zoffset = 8.0f, step = -0.1f; zoffset <= 8.f; zoffset += step) {
        if (zoffset < -60.f) step *= -1.f;
        cameraMatrix.emplace_back(-zoffset);
    }
    //used to index the camera matrix in the loop
    int cameraIndex = 0;
    int cameraSize = cameraMatrix.size();

    auto start = std::chrono::high_resolution_clock::now();
    std::chrono::time_point<std::chrono::high_resolution_clock> end;
    int cycle = 0;

    // Main rendering loop
    while (running) {
        renderer.canvas.checkInput();
        renderer.clear();

        // Rotate the first two cubes in the scene
        scene[0].world = scene[0].world * matrix::makeRotateXYZ(0.1f, 0.1f, 0.0f);
        scene[1].world = scene[1].world * matrix::makeRotateXYZ(0.0f, 0.1f, 0.2f);

        if (renderer.canvas.keyPressed(VK_ESCAPE)) break;

        camera.setTranslationZ(cameraMatrix[cameraIndex]);// Update camera position

        for (auto& m : scene)
            render1(renderer, m, camera, L);
        renderer.present();

        if (++cameraIndex == cameraSize) {
            end = std::chrono::high_resolution_clock::now();
            std::cout << ++cycle << " :" << std::chrono::duration<double, std::milli>(end - start).count() << "ms\n";
            start = std::chrono::high_resolution_clock::now();
            cameraIndex = 0;
        }
    }
}

//Using Mesh instead of Mesh*
//Using render2 (Passing Mesh by reference)+(Memorization of transformation)
void scene1_3() {
    Renderer renderer;
    matrix camera;
    Light L{ vec4(0.f, 1.f, 1.f, 0.f), colour(1.0f, 1.0f, 1.0f), colour(0.1f, 0.1f, 0.1f) };
    bool running = true;

    std::vector<Mesh> scene(40);

    for (unsigned int i = 0; i < 20; i++) {
        scene[i * 2] = Mesh::makeCube(1.f);
        scene[i * 2].world = matrix::makeTranslation(-2.0f, 0.0f, (-3 * static_cast<float>(i))) * makeRandomRotation();
        scene[i * 2 + 1] = Mesh::makeCube(1.f);
        scene[i * 2 + 1].world = matrix::makeTranslation(2.0f, 0.0f, (-3 * static_cast<float>(i))) * makeRandomRotation();
    }

    float zoffset = 8.0f; // Initial camera Z-offset
    float step = -0.1f;  // Step size for camera movement

    auto start = std::chrono::high_resolution_clock::now();
    std::chrono::time_point<std::chrono::high_resolution_clock> end;
    int cycle = 0;

    // Main rendering loop
    while (running) {
        renderer.canvas.checkInput();
        renderer.clear();

        camera = matrix::makeTranslation(0, 0, -zoffset); // Update camera position

        // Rotate the first two cubes in the scene
        scene[0].world = scene[0].world * matrix::makeRotateXYZ(0.1f, 0.1f, 0.0f);
        scene[1].world = scene[1].world * matrix::makeRotateXYZ(0.0f, 0.1f, 0.2f);

        if (renderer.canvas.keyPressed(VK_ESCAPE)) break;

        zoffset += step;
        if (zoffset < -60.f || zoffset > 8.f) {
            step *= -1.f;
            if (++cycle % 2 == 0) {
                end = std::chrono::high_resolution_clock::now();
                std::cout << cycle / 2 << " :" << std::chrono::duration<double, std::milli>(end - start).count() << "ms\n";
                start = std::chrono::high_resolution_clock::now();
            }
        }

        for (auto& m : scene)
            render2(renderer, m, camera, L);
        renderer.present();
    }

}
//Using Mesh instead of Mesh*
//Using render3 (Passing Mesh by reference)+(Optimizing algorithm for rasterization)
void scene1_4() {
    Renderer renderer;
    matrix camera;
    Light L{ vec4(0.f, 1.f, 1.f, 0.f), colour(1.0f, 1.0f, 1.0f), colour(0.1f, 0.1f, 0.1f) };
    L.omega_i.normalise();
    bool running = true;

    std::vector<Mesh> scene(40);

    for (unsigned int i = 0; i < 20; i++) {
        scene[i * 2] = Mesh::makeCube(1.f);
        scene[i * 2].world = matrix::makeTranslation(-2.0f, 0.0f, (-3 * static_cast<float>(i))) * makeRandomRotation();
        scene[i * 2 + 1] = Mesh::makeCube(1.f);
        scene[i * 2 + 1].world = matrix::makeTranslation(2.0f, 0.0f, (-3 * static_cast<float>(i))) * makeRandomRotation();
    }

    float zoffset = 8.0f; // Initial camera Z-offset
    float step = -0.1f;  // Step size for camera movement

    auto start = std::chrono::high_resolution_clock::now();
    std::chrono::time_point<std::chrono::high_resolution_clock> end;
    int cycle = 0;

    // Main rendering loop
    while (running) {
        renderer.canvas.checkInput();
        renderer.clear();

        camera = matrix::makeTranslation(0, 0, -zoffset); // Update camera position

        // Rotate the first two cubes in the scene
        scene[0].world = scene[0].world * matrix::makeRotateXYZ(0.1f, 0.1f, 0.0f);
        scene[1].world = scene[1].world * matrix::makeRotateXYZ(0.0f, 0.1f, 0.2f);

        if (renderer.canvas.keyPressed(VK_ESCAPE)) break;

        zoffset += step;
        if (zoffset < -60.f || zoffset > 8.f) {
            step *= -1.f;
            if (++cycle % 2 == 0) {
                end = std::chrono::high_resolution_clock::now();
                std::cout << cycle / 2 << " :" << std::chrono::duration<double, std::milli>(end - start).count() << "ms\n";
                start = std::chrono::high_resolution_clock::now();
            }
        }

        for (auto& m : scene)
            render3(renderer, m, camera, L);
        renderer.present();
    }
}
//renderer4 (backface culling)
void scene1_5() {
    Renderer renderer;
    matrix camera;
    Light L{ vec4(0.f, 1.f, 1.f, 0.f), colour(1.0f, 1.0f, 1.0f), colour(0.1f, 0.1f, 0.1f) };
    L.omega_i.normalise();
    bool running = true;

    std::vector<Mesh> scene(40);

    for (unsigned int i = 0; i < 20; i++) {
        scene[i * 2] = Mesh::makeCube(1.f);
        scene[i * 2].world = matrix::makeTranslation(-2.0f, 0.0f, (-3 * static_cast<float>(i))) * makeRandomRotation();
        scene[i * 2 + 1] = Mesh::makeCube(1.f);
        scene[i * 2 + 1].world = matrix::makeTranslation(2.0f, 0.0f, (-3 * static_cast<float>(i))) * makeRandomRotation();
    }

    float zoffset = 8.0f; // Initial camera Z-offset
    float step = -0.1f;  // Step size for camera movement

    auto start = std::chrono::high_resolution_clock::now();
    std::chrono::time_point<std::chrono::high_resolution_clock> end;
    int cycle = 0;

    // Main rendering loop
    while (running) {
        renderer.canvas.checkInput();
        renderer.clear();

        camera = matrix::makeTranslation(0, 0, -zoffset); // Update camera position

        // Rotate the first two cubes in the scene
        scene[0].world = scene[0].world * matrix::makeRotateXYZ(0.1f, 0.1f, 0.0f);
        scene[1].world = scene[1].world * matrix::makeRotateXYZ(0.0f, 0.1f, 0.2f);

        if (renderer.canvas.keyPressed(VK_ESCAPE)) break;

        zoffset += step;
        if (zoffset < -60.f || zoffset > 8.f) {
            step *= -1.f;
            if (++cycle % 2 == 0) {
                end = std::chrono::high_resolution_clock::now();
                std::cout << cycle / 2 << " :" << std::chrono::duration<double, std::milli>(end - start).count() << "ms\n";
                start = std::chrono::high_resolution_clock::now();
            }
        }

        for (auto& m : scene)
            render4(renderer, m, camera, L);
        renderer.present();
    }
}
//renderer5 SIMD
void scene1_6() {
    Renderer renderer;
    matrix camera;
    Light L{ vec4(0.f, 1.f, 1.f, 0.f), colour(1.0f, 1.0f, 1.0f), colour(0.1f, 0.1f, 0.1f) };
    L.omega_i.normalise();
    bool running = true;

    std::vector<Mesh> scene(40);

    for (unsigned int i = 0; i < 20; i++) {
        scene[i * 2] = Mesh::makeCube(1.f);
        scene[i * 2].world = matrix::makeTranslation(-2.0f, 0.0f, (-3 * static_cast<float>(i))) * makeRandomRotation();
        scene[i * 2 + 1] = Mesh::makeCube(1.f);
        scene[i * 2 + 1].world = matrix::makeTranslation(2.0f, 0.0f, (-3 * static_cast<float>(i))) * makeRandomRotation();
    }

    float zoffset = 8.0f; // Initial camera Z-offset
    float step = -0.1f;  // Step size for camera movement

    auto start = std::chrono::high_resolution_clock::now();
    std::chrono::time_point<std::chrono::high_resolution_clock> end;
    int cycle = 0;

    // Main rendering loop
    while (running) {
        renderer.canvas.checkInput();
        renderer.clear();

        camera = matrix::makeTranslation(0, 0, -zoffset); // Update camera position

        // Rotate the first two cubes in the scene
        scene[0].world = scene[0].world * matrix::makeRotateXYZ(0.1f, 0.1f, 0.0f);
        scene[1].world = scene[1].world * matrix::makeRotateXYZ(0.0f, 0.1f, 0.2f);

        if (renderer.canvas.keyPressed(VK_ESCAPE)) break;

        zoffset += step;
        if (zoffset < -60.f || zoffset > 8.f) {
            step *= -1.f;
            if (++cycle % 2 == 0) {
                end = std::chrono::high_resolution_clock::now();
                std::cout << cycle / 2 << " :" << std::chrono::duration<double, std::milli>(end - start).count() << "ms\n";
                start = std::chrono::high_resolution_clock::now();
            }
        }

        for (auto& m : scene)
            render5(renderer, m, camera, L);
        renderer.present();
    }
}
//MultiThreading for mesh
void scene1_7() {
    Renderer renderer;
    matrix camera;
    Light L{ vec4(0.f, 1.f, 1.f, 0.f), colour(1.0f, 1.0f, 1.0f), colour(0.1f, 0.1f, 0.1f) };
    L.omega_i.normalise();
    bool running = true;

    std::vector<Mesh> scene(40);

    for (unsigned int i = 0; i < 20; i++) {
        scene[i * 2] = Mesh::makeCube(1.f);
        scene[i * 2].world = matrix::makeTranslation(-2.0f, 0.0f, (-3 * static_cast<float>(i))) * makeRandomRotation();
        scene[i * 2 + 1] = Mesh::makeCube(1.f);
        scene[i * 2 + 1].world = matrix::makeTranslation(2.0f, 0.0f, (-3 * static_cast<float>(i))) * makeRandomRotation();
    }

    float zoffset = 8.0f;
    float step = -0.1f;

    auto start = std::chrono::high_resolution_clock::now();
    std::chrono::time_point<std::chrono::high_resolution_clock> end;
    int cycle = 0;

    ThreadPool pool;

    while (running) {
        renderer.canvas.checkInput();
        renderer.clear();

        camera = matrix::makeTranslation(0, 0, -zoffset);
        scene[0].world = scene[0].world * matrix::makeRotateXYZ(0.1f, 0.1f, 0.0f);
        scene[1].world = scene[1].world * matrix::makeRotateXYZ(0.0f, 0.1f, 0.2f);

        if (renderer.canvas.keyPressed(VK_ESCAPE)) break;

        zoffset += step;
        if (zoffset < -60.f || zoffset > 8.f) {
            step *= -1.f;
            if (++cycle % 2 == 0) {
                end = std::chrono::high_resolution_clock::now();
                std::cout << cycle / 2 << " :" << std::chrono::duration<double, std::milli>(end - start).count() << "ms\n";
                start = std::chrono::high_resolution_clock::now();
            }
        }


        for (auto& m : scene) {
            pool.enqueue(render5, std::ref(renderer), std::ref(m), std::ref(camera), std::ref(L));
        }
        pool.waitForAllTasks();
        renderer.present();
    }
}


// Scene with a grid of cubes and a moving sphere
// No input variables
void scene2() {
    Renderer renderer;
    matrix camera = matrix::makeIdentity();
    Light L{ vec4(0.f, 1.f, 1.f, 0.f), colour(1.0f, 1.0f, 1.0f), colour(0.1f, 0.1f, 0.1f) };

    std::vector<Mesh*> scene;

    struct rRot { float x; float y; float z; }; // Structure to store random rotation parameters
    std::vector<rRot> rotations;

    RandomNumberGenerator& rng = RandomNumberGenerator::getInstance();

    // Create a grid of cubes with random rotations
    for (unsigned int y = 0; y < 6; y++) {
        for (unsigned int x = 0; x < 8; x++) {
            Mesh* m = new Mesh();
            *m = Mesh::makeCube(1.f);
            scene.push_back(m);
            m->world = matrix::makeTranslation(-7.0f + (static_cast<float>(x) * 2.f), 5.0f - (static_cast<float>(y) * 2.f), -8.f);
            rRot r{ rng.getRandomFloat(-.1f, .1f), rng.getRandomFloat(-.1f, .1f), rng.getRandomFloat(-.1f, .1f) };
            rotations.push_back(r);
        }
    }

    // Create a sphere and add it to the scene
    Mesh* sphere = new Mesh();
    *sphere = Mesh::makeSphere(1.0f, 10, 20);
    scene.push_back(sphere);
    float sphereOffset = -6.f;
    float sphereStep = 0.1f;
    sphere->world = matrix::makeTranslation(sphereOffset, 0.f, -6.f);

    auto start = std::chrono::high_resolution_clock::now();
    std::chrono::time_point<std::chrono::high_resolution_clock> end;
    int cycle = 0;

    bool running = true;
    while (running) {
        renderer.canvas.checkInput();
        renderer.clear();

        // Rotate each cube in the grid
        for (unsigned int i = 0; i < rotations.size(); i++)
            scene[i]->world = scene[i]->world * matrix::makeRotateXYZ(rotations[i].x, rotations[i].y, rotations[i].z);

        // Move the sphere back and forth
        sphereOffset += sphereStep;
        sphere->world = matrix::makeTranslation(sphereOffset, 0.f, -6.f);
        if (sphereOffset > 6.0f || sphereOffset < -6.0f) {
            sphereStep *= -1.f;
            if (++cycle % 2 == 0) {
                end = std::chrono::high_resolution_clock::now();
                std::cout << cycle / 2 << " :" << std::chrono::duration<double, std::milli>(end - start).count() << "ms\n";
                start = std::chrono::high_resolution_clock::now();
            }
        }

        if (renderer.canvas.keyPressed(VK_ESCAPE)) break;

        for (auto& m : scene)
            render(renderer, m, camera, L);
        renderer.present();
    }

    for (auto& m : scene)
        delete m;
}
void scene2_4() {
    Renderer renderer;
    matrix camera = matrix::makeIdentity();
    Light L{ vec4(0.f, 1.f, 1.f, 0.f), colour(1.0f, 1.0f, 1.0f), colour(0.1f, 0.1f, 0.1f) };
	L.omega_i.normalise();

    std::vector<Mesh> scene(49);

    struct rRot { float x; float y; float z; }; // Structure to store random rotation parameters
    std::vector<rRot> rotations;

    RandomNumberGenerator& rng = RandomNumberGenerator::getInstance();

    int index;
    // Create a grid of cubes with random rotations
    for (unsigned int y = 0; y < 6; y++) {
        for (unsigned int x = 0; x < 8; x++) {
			index = y * 8 + x;
            scene[index] = Mesh::makeCube(1.f);
            scene[index].world = matrix::makeTranslation(-7.0f + (static_cast<float>(x) * 2.f), 5.0f - (static_cast<float>(y) * 2.f), -8.f);
            rRot r{ rng.getRandomFloat(-.1f, .1f), rng.getRandomFloat(-.1f, .1f), rng.getRandomFloat(-.1f, .1f) };
            rotations.push_back(r);
        }
    }

    // Create a sphere and add it to the scene
    scene[48]= Mesh::makeSphere(1.0f, 10, 20);
    float sphereOffset = -6.f;
    float sphereStep = 0.1f;
    scene[48].world = matrix::makeTranslation(sphereOffset, 0.f, -6.f);

    auto start = std::chrono::high_resolution_clock::now();
    std::chrono::time_point<std::chrono::high_resolution_clock> end;
    int cycle = 0;

    bool running = true;
    while (running) {
        renderer.canvas.checkInput();
        renderer.clear();

        // Rotate each cube in the grid
        for (unsigned int i = 0; i < rotations.size(); i++)
            scene[i].world = scene[i].world * matrix::makeRotateXYZ(rotations[i].x, rotations[i].y, rotations[i].z);

        // Move the sphere back and forth
        sphereOffset += sphereStep;
        scene[48].world = matrix::makeTranslation(sphereOffset, 0.f, -6.f);
        if (sphereOffset > 6.0f || sphereOffset < -6.0f) {
            sphereStep *= -1.f;
            if (++cycle % 2 == 0) {
                end = std::chrono::high_resolution_clock::now();
                std::cout << cycle / 2 << " :" << std::chrono::duration<double, std::milli>(end - start).count() << "ms\n";
                start = std::chrono::high_resolution_clock::now();
            }
        }

        if (renderer.canvas.keyPressed(VK_ESCAPE)) break;

        for (auto& m : scene)
            render3(renderer, m, camera, L);
        renderer.present();
    }
}
void scene2_5() {
    Renderer renderer;
    matrix camera = matrix::makeIdentity();
    Light L{ vec4(0.f, 1.f, 1.f, 0.f), colour(1.0f, 1.0f, 1.0f), colour(0.1f, 0.1f, 0.1f) };
    L.omega_i.normalise();

    std::vector<Mesh> scene(49);

    struct rRot { float x; float y; float z; }; // Structure to store random rotation parameters
    std::vector<rRot> rotations;

    RandomNumberGenerator& rng = RandomNumberGenerator::getInstance();

    int index;
    // Create a grid of cubes with random rotations
    for (unsigned int y = 0; y < 6; y++) {
        for (unsigned int x = 0; x < 8; x++) {
            index = y * 8 + x;
            scene[index] = Mesh::makeCube(1.f);
            scene[index].world = matrix::makeTranslation(-7.0f + (static_cast<float>(x) * 2.f), 5.0f - (static_cast<float>(y) * 2.f), -8.f);
            rRot r{ rng.getRandomFloat(-.1f, .1f), rng.getRandomFloat(-.1f, .1f), rng.getRandomFloat(-.1f, .1f) };
            rotations.push_back(r);
        }
    }

    // Create a sphere and add it to the scene
    scene[48] = Mesh::makeSphere(1.0f, 10, 20);
    float sphereOffset = -6.f;
    float sphereStep = 0.1f;
    scene[48].world = matrix::makeTranslation(sphereOffset, 0.f, -6.f);

    auto start = std::chrono::high_resolution_clock::now();
    std::chrono::time_point<std::chrono::high_resolution_clock> end;
    int cycle = 0;

    bool running = true;
    while (running) {
        renderer.canvas.checkInput();
        renderer.clear();

        // Rotate each cube in the grid
        for (unsigned int i = 0; i < rotations.size(); i++)
            scene[i].world = scene[i].world * matrix::makeRotateXYZ(rotations[i].x, rotations[i].y, rotations[i].z);

        // Move the sphere back and forth
        sphereOffset += sphereStep;
        scene[48].world = matrix::makeTranslation(sphereOffset, 0.f, -6.f);
        if (sphereOffset > 6.0f || sphereOffset < -6.0f) {
            sphereStep *= -1.f;
            if (++cycle % 2 == 0) {
                end = std::chrono::high_resolution_clock::now();
                std::cout << cycle / 2 << " :" << std::chrono::duration<double, std::milli>(end - start).count() << "ms\n";
                start = std::chrono::high_resolution_clock::now();
            }
        }

        if (renderer.canvas.keyPressed(VK_ESCAPE)) break;

        for (auto& m : scene)
            render4(renderer, m, camera, L);
        renderer.present();
    }
}
void scene2_6() {
    Renderer renderer;
    matrix camera = matrix::makeIdentity();
    Light L{ vec4(0.f, 1.f, 1.f, 0.f), colour(1.0f, 1.0f, 1.0f), colour(0.1f, 0.1f, 0.1f) };
    L.omega_i.normalise();

    std::vector<Mesh> scene(49);

    struct rRot { float x; float y; float z; }; // Structure to store random rotation parameters
    std::vector<rRot> rotations;

    RandomNumberGenerator& rng = RandomNumberGenerator::getInstance();

    int index;
    // Create a grid of cubes with random rotations
    for (unsigned int y = 0; y < 6; y++) {
        for (unsigned int x = 0; x < 8; x++) {
            index = y * 8 + x;
            scene[index] = Mesh::makeCube(1.f);
            scene[index].world = matrix::makeTranslation(-7.0f + (static_cast<float>(x) * 2.f), 5.0f - (static_cast<float>(y) * 2.f), -8.f);
            rRot r{ rng.getRandomFloat(-.1f, .1f), rng.getRandomFloat(-.1f, .1f), rng.getRandomFloat(-.1f, .1f) };
            rotations.push_back(r);
        }
    }

    // Create a sphere and add it to the scene
    scene[48] = Mesh::makeSphere(1.0f, 10, 20);
    float sphereOffset = -6.f;
    float sphereStep = 0.1f;
    scene[48].world = matrix::makeTranslation(sphereOffset, 0.f, -6.f);

    auto start = std::chrono::high_resolution_clock::now();
    std::chrono::time_point<std::chrono::high_resolution_clock> end;
    int cycle = 0;

    bool running = true;
    while (running) {
        renderer.canvas.checkInput();
        renderer.clear();

        // Rotate each cube in the grid
        for (unsigned int i = 0; i < rotations.size(); i++)
            scene[i].world = scene[i].world * matrix::makeRotateXYZ(rotations[i].x, rotations[i].y, rotations[i].z);

        // Move the sphere back and forth
        sphereOffset += sphereStep;
        scene[48].world = matrix::makeTranslation(sphereOffset, 0.f, -6.f);
        if (sphereOffset > 6.0f || sphereOffset < -6.0f) {
            sphereStep *= -1.f;
            if (++cycle % 2 == 0) {
                end = std::chrono::high_resolution_clock::now();
                std::cout << cycle / 2 << " :" << std::chrono::duration<double, std::milli>(end - start).count() << "ms\n";
                start = std::chrono::high_resolution_clock::now();
            }
        }

        if (renderer.canvas.keyPressed(VK_ESCAPE)) break;

        for (auto& m : scene)
            render5(renderer, m, camera, L);
        renderer.present();
    }
}
void scene2_7() {
    Renderer renderer;
    matrix camera = matrix::makeIdentity();
    Light L{ vec4(0.f, 1.f, 1.f, 0.f), colour(1.0f, 1.0f, 1.0f), colour(0.1f, 0.1f, 0.1f) };
    L.omega_i.normalise();

    std::vector<Mesh> scene(49);

    struct rRot { float x; float y; float z; }; // Structure to store random rotation parameters
    std::vector<rRot> rotations;

    RandomNumberGenerator& rng = RandomNumberGenerator::getInstance();

    int index;
    // Create a grid of cubes with random rotations
    for (unsigned int y = 0; y < 6; y++) {
        for (unsigned int x = 0; x < 8; x++) {
            index = y * 8 + x;
            scene[index] = Mesh::makeCube(1.f);
            scene[index].world = matrix::makeTranslation(-7.0f + (static_cast<float>(x) * 2.f), 5.0f - (static_cast<float>(y) * 2.f), -8.f);
            rRot r{ rng.getRandomFloat(-.1f, .1f), rng.getRandomFloat(-.1f, .1f), rng.getRandomFloat(-.1f, .1f) };
            rotations.push_back(r);
        }
    }

    // Create a sphere and add it to the scene
    scene[48] = Mesh::makeSphere(1.0f, 10, 20);
    float sphereOffset = -6.f;
    float sphereStep = 0.1f;
    scene[48].world = matrix::makeTranslation(sphereOffset, 0.f, -6.f);

    auto start = std::chrono::high_resolution_clock::now();
    std::chrono::time_point<std::chrono::high_resolution_clock> end;
    int cycle = 0;

	ThreadPool pool;
    bool running = true;
    while (running) {
        renderer.canvas.checkInput();
        renderer.clear();

        // Rotate each cube in the grid
        for (unsigned int i = 0; i < rotations.size(); i++)
            scene[i].world = scene[i].world * matrix::makeRotateXYZ(rotations[i].x, rotations[i].y, rotations[i].z);

        // Move the sphere back and forth
        sphereOffset += sphereStep;
        scene[48].world = matrix::makeTranslation(sphereOffset, 0.f, -6.f);
        if (sphereOffset > 6.0f || sphereOffset < -6.0f) {
            sphereStep *= -1.f;
            if (++cycle % 2 == 0) {
                end = std::chrono::high_resolution_clock::now();
                std::cout << cycle / 2 << " :" << std::chrono::duration<double, std::milli>(end - start).count() << "ms\n";
                start = std::chrono::high_resolution_clock::now();
            }
        }

        if (renderer.canvas.keyPressed(VK_ESCAPE)) break;

        for (auto& m : scene) {
            pool.enqueue(render5, std::ref(renderer), std::ref(m), std::ref(camera), std::ref(L));
        }
        pool.waitForAllTasks();
        renderer.present();
    }
}

void scene3_1_SingleThread() {
    Renderer renderer;
    matrix camera;
    Light L{ vec4(0.f, 1.f, 1.f, 0.f), colour(1.0f, 1.0f, 1.0f), colour(0.1f, 0.1f, 0.1f) };
    L.omega_i.normalise();
    bool running = true;

    std::vector<Mesh> scene(40);

    for (unsigned int i = 0; i < 40; i++) {
        scene[i] = Mesh::makeCube(1.f);
        scene[i].world = matrix::makeTranslation(0.0f, 0.0f, (-3 * static_cast<float>(i))-5) * makeRandomRotation();
    }

    float zoffset = 0.8f; // Initial camera Z-offset
    float step = -0.01f;  // Step size for camera movement

    auto start = std::chrono::high_resolution_clock::now();
    std::chrono::time_point<std::chrono::high_resolution_clock> end;
    int cycle = 0;

    // Main rendering loop
    while (running) {
        renderer.canvas.checkInput();
        renderer.clear();

        camera = matrix::makeTranslation(0, 0, -zoffset); // Update camera position

        // Rotate the first two cubes in the scene
        scene[0].world = scene[0].world * matrix::makeRotateXYZ(0.1f, 0.1f, 0.0f);
        scene[1].world = scene[1].world * matrix::makeRotateXYZ(0.0f, 0.1f, 0.2f);

        if (renderer.canvas.keyPressed(VK_ESCAPE)) break;

        zoffset += step;
        if (zoffset < -6.f || zoffset > 0.8f) {
            step *= -1.f;
            if (++cycle % 2 == 0) {
                end = std::chrono::high_resolution_clock::now();
                std::cout << cycle / 2 << " :" << std::chrono::duration<double, std::milli>(end - start).count() << "ms\n";
                start = std::chrono::high_resolution_clock::now();
            }
        }

        for (auto& m : scene)
            render5(renderer, m, camera, L);
        renderer.present();
    }
}
void scene3_2_SingleThread() {
    Renderer renderer;
    matrix camera;
    Light L{ vec4(0.f, 1.f, 1.f, 0.f), colour(1.0f, 1.0f, 1.0f), colour(0.1f, 0.1f, 0.1f) };
    L.omega_i.normalise();
    bool running = true;

    std::vector<Mesh> scene(40);

    for (int i = 39; i >= 0; i--) {
        scene[i] = Mesh::makeCube(1.f);
        scene[i].world = matrix::makeTranslation(0.0f, 0.0f, (-3 * static_cast<float>(i))-5) * makeRandomRotation();
    }

    float zoffset = 0.8f; // Initial camera Z-offset
    float step = -0.01f;  // Step size for camera movement

    auto start = std::chrono::high_resolution_clock::now();
    std::chrono::time_point<std::chrono::high_resolution_clock> end;
    int cycle = 0;

    // Main rendering loop
    while (running) {
        renderer.canvas.checkInput();
        renderer.clear();

        camera = matrix::makeTranslation(0, 0, -zoffset); // Update camera position

        // Rotate the first two cubes in the scene
        scene[0].world = scene[0].world * matrix::makeRotateXYZ(0.1f, 0.1f, 0.0f);
        scene[1].world = scene[1].world * matrix::makeRotateXYZ(0.0f, 0.1f, 0.2f);

        if (renderer.canvas.keyPressed(VK_ESCAPE)) break;

        zoffset += step;
        if (zoffset < -6.f || zoffset > 0.8f) {
            step *= -1.f;
            if (++cycle % 2 == 0) {
                end = std::chrono::high_resolution_clock::now();
                std::cout << cycle / 2 << " :" << std::chrono::duration<double, std::milli>(end - start).count() << "ms\n";
                start = std::chrono::high_resolution_clock::now();
            }
        }

        for (auto& m : scene)
            render5(renderer, m, camera, L);
        renderer.present();
    }
}
void scene3_1_MultiThread() {
    Renderer renderer;
    matrix camera;
    Light L{ vec4(0.f, 1.f, 1.f, 0.f), colour(1.0f, 1.0f, 1.0f), colour(0.1f, 0.1f, 0.1f) };
    L.omega_i.normalise();
    bool running = true;

    std::vector<Mesh> scene(40);

    for (unsigned int i = 0; i < 40; i++) {
        scene[i] = Mesh::makeCube(1.f);
        scene[i].world = matrix::makeTranslation(0.0f, 0.0f, (-3 * static_cast<float>(i)) - 5) * makeRandomRotation();
    }

    float zoffset = 0.8f; // Initial camera Z-offset
    float step = -0.01f;  // Step size for camera movement

    auto start = std::chrono::high_resolution_clock::now();
    std::chrono::time_point<std::chrono::high_resolution_clock> end;
    int cycle = 0;

    ThreadPool pool;

    while (running) {
        renderer.canvas.checkInput();
        renderer.clear();

        camera = matrix::makeTranslation(0, 0, -zoffset);
        scene[0].world = scene[0].world * matrix::makeRotateXYZ(0.1f, 0.1f, 0.0f);
        scene[1].world = scene[1].world * matrix::makeRotateXYZ(0.0f, 0.1f, 0.2f);

        if (renderer.canvas.keyPressed(VK_ESCAPE)) break;

        zoffset += step;
        if (zoffset < -6.f || zoffset > 0.8f) {
            step *= -1.f;
            if (++cycle % 2 == 0) {
                end = std::chrono::high_resolution_clock::now();
                std::cout << cycle / 2 << " :" << std::chrono::duration<double, std::milli>(end - start).count() << "ms\n";
                start = std::chrono::high_resolution_clock::now();
            }
        }


        for (auto& m : scene) {
            pool.enqueue(render5, std::ref(renderer), std::ref(m), std::ref(camera), std::ref(L));
        }
        pool.waitForAllTasks();
        renderer.present();
    }
}
void scene3_2_MultiThread() {
    Renderer renderer;
    matrix camera;
    Light L{ vec4(0.f, 1.f, 1.f, 0.f), colour(1.0f, 1.0f, 1.0f), colour(0.1f, 0.1f, 0.1f) };
    L.omega_i.normalise();
    bool running = true;

    std::vector<Mesh> scene(40);

    for (int i = 39; i >= 0; i--) {
        scene[i] = Mesh::makeCube(1.f);
        scene[i].world = matrix::makeTranslation(0.0f, 0.0f, (-3 * static_cast<float>(i)) - 5) * makeRandomRotation();
    }

    float zoffset = 0.8f; // Initial camera Z-offset
    float step = -0.01f;  // Step size for camera movement

    auto start = std::chrono::high_resolution_clock::now();
    std::chrono::time_point<std::chrono::high_resolution_clock> end;
    int cycle = 0;

    ThreadPool pool;

    while (running) {
        renderer.canvas.checkInput();
        renderer.clear();

        camera = matrix::makeTranslation(0, 0, -zoffset);
        scene[0].world = scene[0].world * matrix::makeRotateXYZ(0.1f, 0.1f, 0.0f);
        scene[1].world = scene[1].world * matrix::makeRotateXYZ(0.0f, 0.1f, 0.2f);

        if (renderer.canvas.keyPressed(VK_ESCAPE)) break;

        zoffset += step;
        if (zoffset < -6.f || zoffset > 0.8f) {
            step *= -1.f;
            if (++cycle % 2 == 0) {
                end = std::chrono::high_resolution_clock::now();
                std::cout << cycle / 2 << " :" << std::chrono::duration<double, std::milli>(end - start).count() << "ms\n";
                start = std::chrono::high_resolution_clock::now();
            }
        }


        for (auto& m : scene) {
            pool.enqueue(render5, std::ref(renderer), std::ref(m), std::ref(camera), std::ref(L));
        }
        pool.waitForAllTasks();
        renderer.present();
    }
}
// Entry point of the application
// No input variables
int main() {
	//All of the scenes are listed below 
    
    //scene1();
	//scene1_1();
	//scene1_2();
	//scene1_3();
	//scene1_4();
	//scene1_5();
	//scene1_6();
	//scene1_7();
	
    //scene2();
	//scene2_4();
	//scene2_5();
	//scene2_6();
	//scene2_7();
	//scene3_1_SingleThread();
	//scene3_2_SingleThread();
	//scene3_1_MultiThread();
	scene3_2_MultiThread();

    return 0;
}