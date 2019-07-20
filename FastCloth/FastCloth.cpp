// FastCloth.cpp : This file contains the 'main' function. Program execution
// begins and ends there.
//

#include <immintrin.h>
#include <chrono>
#include <iostream>
#include <thread>
#include "Constants.h"
#include "CpuInfo.cpp"

#define USE_AVX
#define THREAD_NUM 1

#ifdef USE_AVX
#define CALL_KERNEL(ker) ker##_avx
#else
#define CALL_KERNEL(ker) ker
#endif

#define GET_STRIDE8(arr, i)                                              \
  _mm256_setr_ps(arr[i], arr[i + 1], arr[i + 2], arr[i + 3], arr[i + 4], \
                 arr[i + 5], arr[i + 6], arr[i + 7])

const InstructionSet::InstructionSet_Internal InstructionSet::CPU_Rep;

/*
        COORIDANTE SYSTEM
        Y
        ^
        |
        |
        x---------->X
        Z

                ..
                NX | NX+1|     |  ..  |2NX-1
                0  |  1  |  2  |  ..  | NX-1
*/

/*
        POS: Rx0 Ry0 Rz0 Rx1 Ry1 Rz1 ...
        VEL: Vx0 Vy0 Vz0 Vx1 Vy1 Vz1 ...
        ACC: Ax0 Ay0 Az0 Ax1 Ay1 Az1 ...
        X : 3i
        Y : 3i + 1
        Z : 3i +2
                len (pos) = 3*NX*NY
                256 = 8 * 32
*/

void init_pos(float* pos) {
  for (auto i = 0; i < Constants::nx; i++) {
    for (auto j = 0; j < Constants::ny; j++) {
      auto idx = i + Constants::nx * j;
      pos[idx * 3] = (float)i * Constants::lfree;
      pos[idx * 3 + 1] = (float)j * Constants::lfree;
      pos[idx * 3 + 2] = 0.0f;
    }
  }
}

// stride should be N*3 for single thread
// NX = X + V*dt + ACC*(dt*dt*0.5)
void kernel_pos_avx(const float* pos, const float* vel, const float* acc,
                    float* npos, const int stride) {
  const __m256 dt = _mm256_set1_ps(Constants::dt);
  const __m256 half_dt2 = _mm256_set1_ps(Constants::dt * Constants::dt * 0.5f);
  for (auto i = 0; i < stride / 8; i++) {
    auto idx = i * 8;
    __debugbreak();
    const __m256 v = GET_STRIDE8(vel, idx);
    const __m256 a = GET_STRIDE8(acc, idx);
    __m256 x = GET_STRIDE8(pos, idx);
    x = _mm256_fmadd_ps(dt, v, x);
    x = _mm256_fmadd_ps(half_dt2, a, x);
    _mm256_store_ps(npos + idx, x);
  }
}

void kernel_pos(const float* pos, const float* vel, const float* acc,
                float* npos, const int stride) {
  const float dt = Constants::dt;
  const float half_dt2 = dt * 0.5f;
  for (auto i = 0; i < stride; i++) {
    npos[i] = pos[i] + dt * vel[i] + half_dt2 * acc[i];
  }
}

// ADD HERE!
void kernel_vel_avx(const float* vel, const float* acc, const float* nacc,
                    float* nvel, const int stride) {
  const __m256 half_dt = _mm256_set1_ps(Constants::dt * 0.5f);
  for (auto i = 0; i < stride / 8; i++) {
    auto idx = i * 8;
    const __m256 v = GET_STRIDE8(vel, idx);
    __m256 eff_a = GET_STRIDE8(acc, idx);
    
    __m256 na = GET_STRIDE8(nacc, idx);
    eff_a = _mm256_add_ps(eff_a, na);
    __m256 nv = _mm256_fmadd_ps(half_dt, eff_a, v);
    _mm256_store_ps(nvel + idx, nv);
  }
}

void kernel_vel(const float* vel, const float* acc, const float* nacc,
                float* nvel, const int stride) {
  const float half_dt = Constants::dt * 0.5f;
  for (auto i = 0; i < stride; i++) {
    nvel[i] = vel[i] + half_dt * (acc[i] + nacc[i]);
  }
}

inline float dist(const float* pos, const int i, const int j) {
  auto i_idx = 3 * i;
  auto j_idx = 3 * j;
  auto dx = pos[i_idx] - pos[j_idx];
  auto dy = pos[i_idx + 1] - pos[j_idx + 1];
  auto dz = pos[i_idx + 2] - pos[j_idx + 2];
  return hypot(dz, hypot(dx, dy));
}

// HERE STRIDE IS NUM OF PARTICLES
void kernel_acc(const float* pos, const float* acc, float* nacc,
                const int p_idx, const int stride) {
  const int nx = Constants::nx;
  const int ny = Constants::ny;
  for (auto i = 0; i < stride; i++) {
    const int idx = p_idx + i;
    if (idx % nx != (nx - 1)) {
      const int jdx = idx + 1;
      float dr = dist(pos, idx, jdx);
      float f = Constants::k * (dr - Constants::lfree) / dr;
      for (auto ii = 0; ii < 3; ii++) {
        
        nacc[idx * 3 + ii] += f * (pos[jdx * 3 + ii] - pos[idx * 3 + ii]);
      }
    }
    if (idx % nx != 0) {
      const int jdx = idx - 1;
      float dr = dist(pos, idx, jdx);
      float f = Constants::k * (dr - Constants::lfree) / dr;
      for (auto ii = 0; ii < 3; ii++) {
        nacc[idx * 3 + ii] += f * (pos[jdx * 3 + ii] - pos[idx * 3 + ii]);
      }
    }
    if (idx >= nx) {
      const int jdx = idx - nx;
      float dr = dist(pos, idx, jdx);
      float f = Constants::k * (dr - Constants::lfree) / dr;
      for (auto ii = 0; ii < 3; ii++) {
        nacc[idx * 3 + ii] += f * (pos[jdx * 3 + ii] - pos[idx * 3 + ii]);
      }
    }
    if (idx < nx * (ny - 1)) {
      const int jdx = idx + nx;
      float dr = dist(pos, idx, jdx);
      float f = Constants::k * (dr - Constants::lfree) / dr;
      for (auto ii = 0; ii < 3; ii++) {
        nacc[idx * 3 + ii] += f * (pos[jdx * 3 + ii] - pos[idx * 3 + ii]);
      }
    }
  }
}

void kernel_acc_avx(const float* pos, const float* acc, float* nacc,
                    const int p_idx, const int stride) {
  kernel_acc(pos, acc, nacc, p_idx, stride);
}

int main() {
  int times = 4;
  int iterz = 20;
  int size = Constants::N * 3;
  float* pos = new float[size];
  float* vel = new float[size];
  float* acc = new float[size];
  float* npos = new float[size];
  float* nvel = new float[size];
  float* nacc = new float[size];
  float val = 0.0f;

  init_pos(pos);
  using namespace std::chrono;
  auto time_sum = 0.0;
  for (auto run = 0; run < times; run++) {
    auto t0 = high_resolution_clock::now();
    for (auto iter = 0; iter < iterz; iter++) {
      std::vector<std::thread> threads;
      const auto stride = size / THREAD_NUM;
      for (auto th = 0; th < THREAD_NUM; th++) {
        threads.push_back(std::thread([&, th] {
          CALL_KERNEL(kernel_pos)
          (pos + th * stride, vel + th * stride, acc + th * stride,
           npos + th * stride, stride);
        }));
      }
      for (auto th = 0; th < THREAD_NUM; th++) {
        threads[th].join();
      }
      threads.clear();
      for (auto th = 0; th < THREAD_NUM; th++) {
        const auto stride = Constants::N / THREAD_NUM;
        const auto arr_stride = stride * 3;
        threads.push_back(std::thread([&, th] {
          CALL_KERNEL(kernel_acc)
          (pos , acc , nacc ,
           stride * th, stride);
        }));
      }
      for (auto th = 0; th < THREAD_NUM; th++) {
        threads[th].join();
      }
      threads.clear();
      for (auto th = 0; th < THREAD_NUM; th++) {
        const auto stride = size / THREAD_NUM;
        threads.push_back(std::thread([&, th] {
          CALL_KERNEL(kernel_vel)
          (vel + stride * th, acc + stride * th, nacc + stride * th,
           nvel + stride * th, stride);
        }));
      }
      for (auto th = 0; th < THREAD_NUM; th++) {
        threads[th].join();
      }
      threads.clear();
      //   for (auto iter = 0; iter < iterz; iter++) {
      //     CALL_KERNEL(kernel_pos)(pos,vel,acc,npos,size);
      //     for (auto i = 0; i < size; i++) {
      //       if (i % 3 == 1) nacc[i] = Constants::g;
      //     }
      //     CALL_KERNEL(kernel_acc)(pos,acc,nacc,0,Constants::N);
      //     CALL_KERNEL(kernel_vel)(vel,acc,nacc,nvel,size);
      //     float* swap = pos;
      //     pos = npos;
      //     npos = swap;
      //     swap = vel;
      //     vel = nvel;
      //     nvel = swap;
      //     swap = acc;
      //     acc = nacc;
      //     nacc = swap;
      //}
    }
      auto t1 = high_resolution_clock::now();
      time_sum += duration_cast<duration<double>>(t1 - t0).count();
      std::cout << "RUN TOOK: "
                << duration_cast<duration<double>>(t1 - t0).count()
                << " seconds\n";
    
  }
  std::cout << "AVERGAE RUN: " << time_sum / (double)times << "\n";

  // for (auto i = 0; i < size; i++) {
  //  pos[i] = val;
  //  vel[i] = val + size;
  //  val += 1.0f;
  //}
  // for (auto i = 0; i < 30; i++) {
  //  std::cout << pos[i] << "+" << Constants::dt << "*" << vel[i] << " ==> "
  //            << npos[i]<<"\n";
  //}
  // auto& outstream = std::cout;

  // auto support_message = [&outstream](std::string isa_feature,
  //                                    bool is_supported) {
  //  outstream << isa_feature << (is_supported ? " supported" : " not
  //  supported")
  //            << std::endl;
  //};

  // std::cout << InstructionSet::Vendor() << std::endl;
  // std::cout << InstructionSet::Brand() << std::endl;

  // support_message("AVX512CD", InstructionSet::AVX512CD);

  // std::cout << "Hello World!\n";
  // std::thread t1([] { std::cout << "threader!"; });
  // t1.join();
}
