In Progress
# FastCloth
Simulating cloth physics with verlet integration. I used this to play around with SIMD extensions to Intel processors, and try out different optimizations: 1) Multi-threading 2) AVX with FMAD instructions 3) AVX with separate ADD/MUL instead of FMAD 4) SSE (so XMM registers instead of YMM/ZMM) <br>
As expected on my laptop with a Core i7 7500U (dual core) multi-threading hurts performance, but on my PC with a quad core 4770, going from 1 thread to 2 makes it faster, from 2 to 4 still faster and that is as far the buck goes. 8 threads is stil faster than 1 thread but worse than 2 or 4. <br>
Regarding AVX, AVX to my amusement hurts performance. By default most compilers use XMM and SSE instructions, and it does beat AVX.
