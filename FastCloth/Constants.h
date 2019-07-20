#pragma once
namespace Constants {
static const int nx = 64*32;
static const int ny = 256*64;
static const int N = nx * ny;
static const float lx = 2.0;
static const float ly = lx*(float)ny/(float)nx;
static const float lfree = ly / (float)ny;	// Rest length of the springs
static const float dt = 1.0f;
static const float g = 1.0f;
static const float k = 1.0f;
}