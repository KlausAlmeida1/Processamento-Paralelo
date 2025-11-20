/*
 * Projeto 3 - DGEMM com CUDA
 * Comparação: Naive (Global Mem) vs Tiled (Shared Mem)
 * Dupla: Adrielle e Klaus
 *
 * Compilar: nvcc -O3 -arch=sm_89 cuda.cu -o cuda
 * Executar: ./cuda --runs 5 --sizes 512,1024,2048,4096 --validate
 */

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <string>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <cmath>

using namespace std;

#define TILE_SIZE 32

// Macro para checagem de erros
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// --------------------------------------------------------
// KERNEL 1: NAIVE (Versão Básica - Global Memory)
// --------------------------------------------------------
__global__ void dgemm_naive_kernel(const double* A, const double* B, double* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        double soma = 0.0;
        for (int k = 0; k < N; ++k) {
            soma += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = soma;
    }
}

// --------------------------------------------------------
// KERNEL 2: TILED (Versão Otimizada - Shared Memory)
// --------------------------------------------------------
__global__ void dgemm_tiled_kernel(const double* A, const double* B, double* C, int N) {
    __shared__ double As[TILE_SIZE][TILE_SIZE];
    __shared__ double Bs[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int Row = by * TILE_SIZE + ty;
    int Col = bx * TILE_SIZE + tx;

    double Cvalue = 0.0;

    for (int k = 0; k < (N + TILE_SIZE - 1) / TILE_SIZE; ++k) {
        if (Row < N && (k * TILE_SIZE + tx) < N)
            As[ty][tx] = A[Row * N + k * TILE_SIZE + tx];
        else
            As[ty][tx] = 0.0;

        if (Col < N && (k * TILE_SIZE + ty) < N)
            Bs[ty][tx] = B[(k * TILE_SIZE + ty) * N + Col];
        else
            Bs[ty][tx] = 0.0;

        __syncthreads();

        for (int n = 0; n < TILE_SIZE; ++n)
            Cvalue += As[ty][n] * Bs[n][tx];

        __syncthreads();
    }

    if (Row < N && Col < N)
        C[Row * N + Col] = Cvalue;
}

// --------------------------------------------------------
// Helpers
// --------------------------------------------------------
static void preencherMatriz(double* M, int n, mt19937_64& gen) {
    uniform_real_distribution<double> dist(0.0, 10.0);
    for (long long i = 0; i < 1LL * n * n; i++) M[i] = dist(gen);
}

static uint64_t checksum64(const double* C, int N) {
    const uint64_t* p = reinterpret_cast<const uint64_t*>(C);
    size_t words = (size_t)N * (size_t)N * sizeof(double) / sizeof(uint64_t);
    uint64_t h = 0x9e3779b97f4a7c15ULL;
    for (size_t i = 0; i < words; ++i) {
        uint64_t x = p[i];
        h ^= x + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    }
    return h;
}

// Retorna o erro máximo relativo
static double dgemm_cpu_check_return_max(const double* A, const double* B, const double* C_gpu, int N) {
    // Transpõe B para validação rápida na CPU
    vector<double> Bt(1LL*N*N);
    for(int i=0; i<N; ++i)
        for(int j=0; j<N; ++j) Bt[j*N+i] = B[i*N+j];

    double max_diff = 0.0;
    const double epsilon = 1e-12;

    // Passo de amostragem para N muito grande não demorar eternamente
    // Se precisar de rigor absoluto, mude step para 1
    int step = (N > 2048) ? 10 : 1; 
    
    for (int i = 0; i < N; i += step) {
        for (int j = 0; j < N; j += step) {
            double c_ref = 0.0;
            for (int k = 0; k < N; ++k) {
                c_ref += A[i * N + k] * Bt[j * N + k];
            }
            
            // Fórmula do PDF
            double diff = fabs(c_ref - C_gpu[i * N + j]);
            double rel = diff / (fabs(c_ref) + epsilon);
            
            if (rel > max_diff) max_diff = rel;
        }
    }
    return max_diff;
}

static vector<int> parse_int_list(const string& csv) {
    vector<int> out; string cur; stringstream ss(csv);
    while (getline(ss, cur, ',')) if (!cur.empty()) out.push_back(stoi(cur));
    return out;
}

// --------------------------------------------------------
// MAIN
// --------------------------------------------------------
int main(int argc, char** argv) {
    vector<int> sizes = {512, 1024, 2048, 4096};
    int runs = 5;
    bool validate = false;
    string outFile = "resultados_cuda.txt";
    uint64_t seed = 42;

    for (int i = 1; i < argc; ++i) {
        string a = argv[i];
        if (a == "--runs" && i + 1 < argc) runs = stoi(argv[++i]);
        else if (a == "--sizes" && i + 1 < argc) sizes = parse_int_list(argv[++i]);
        else if (a == "--out" && i + 1 < argc) outFile = argv[++i];
        else if (a == "--validate") validate = true;
        else if (a == "--seed" && i + 1 < argc) seed = stoull(argv[++i]);
    }

    ofstream ofs(outFile);
    if (!ofs) return 1;

    ofs << "DGEMM CUDA (Naive vs Tiled) — Rodadas multiplas\n";
    ofs << "runs=" << runs << ", seed=" << seed << ", GPU=RTX4070\n\n";
    ofs << fixed << setprecision(6);

    mt19937_64 gen(seed);

    for (int N : sizes) {
        size_t bytes = 1LL * N * N * sizeof(double);
        
        vector<double> h_A(1LL*N*N), h_B(1LL*N*N), h_C(1LL*N*N);
        preencherMatriz(h_A.data(), N, gen);
        preencherMatriz(h_B.data(), N, gen);

        double *d_A, *d_B, *d_C;
        cudaCheckError(cudaMalloc(&d_A, bytes));
        cudaCheckError(cudaMalloc(&d_B, bytes));
        cudaCheckError(cudaMalloc(&d_C, bytes));

        cudaCheckError(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));

        dim3 dimBlock(TILE_SIZE, TILE_SIZE);
        dim3 dimGrid((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

        // ==========================================
        // 1. Rodar Versão NAIVE
        // ==========================================
        cout << "Rodando NAIVE N=" << N << "..." << endl;
        dgemm_naive_kernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);
        cudaCheckError(cudaDeviceSynchronize());

        vector<double> t_naive;
        for(int r=0; r<runs; ++r) {
            auto t0 = chrono::high_resolution_clock::now();
            dgemm_naive_kernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);
            cudaCheckError(cudaDeviceSynchronize());
            auto t1 = chrono::high_resolution_clock::now();
            t_naive.push_back(chrono::duration<double>(t1-t0).count());
        }
        
        cudaCheckError(cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost));
        uint64_t chkNaive = checksum64(h_C.data(), N);
        
        double maxDiffNaive = -1.0;
        if(validate) {
            maxDiffNaive = dgemm_cpu_check_return_max(h_A.data(), h_B.data(), h_C.data(), N);
            cout << "  > Naive Validacao: Max Diff = " << maxDiffNaive << endl;
        }

        double meanN = accumulate(t_naive.begin(), t_naive.end(), 0.0)/runs;
        double sdN = 0.0; for(double x:t_naive) sdN += (x-meanN)*(x-meanN); sdN=sqrt(sdN/(runs>1?runs-1:1));
        
        ofs << "N = " << N << ", Kernel = Naive\n";
        ofs << "  checksum = 0x" << hex << chkNaive << dec << "\n";
        if(validate) ofs << "  erro_relativo_max = " << maxDiffNaive << " " << (maxDiffNaive < 1e-8 ? "[OK]" : "[ERRO]") << "\n";
        else ofs << "  erro_relativo_max = N/A (sem --validate)\n";
        ofs << "  tempos: "; for(auto t:t_naive) ofs << t << ", "; ofs << "\n";
        ofs << "  media: " << meanN << " s, desvio: " << sdN << " s\n\n";

        // ==========================================
        // 2. Rodar Versão TILED
        // ==========================================
        cout << "Rodando TILED N=" << N << "..." << endl;
        cudaCheckError(cudaMemset(d_C, 0, bytes));
        dgemm_tiled_kernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);
        cudaCheckError(cudaDeviceSynchronize());

        vector<double> t_tiled;
        for(int r=0; r<runs; ++r) {
            cudaCheckError(cudaMemset(d_C, 0, bytes)); 
            auto t0 = chrono::high_resolution_clock::now();
            dgemm_tiled_kernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);
            cudaCheckError(cudaDeviceSynchronize());
            auto t1 = chrono::high_resolution_clock::now();
            t_tiled.push_back(chrono::duration<double>(t1-t0).count());
        }
        
        cudaCheckError(cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost));
        uint64_t chkTiled = checksum64(h_C.data(), N);

        double maxDiffTiled = -1.0;
        if(validate) {
            maxDiffTiled = dgemm_cpu_check_return_max(h_A.data(), h_B.data(), h_C.data(), N);
            cout << "  > Tiled Validacao: Max Diff = " << maxDiffTiled << endl;
        }
        
        double meanT = accumulate(t_tiled.begin(), t_tiled.end(), 0.0)/runs;
        double sdT = 0.0; for(double x:t_tiled) sdT += (x-meanT)*(x-meanT); sdT=sqrt(sdT/(runs>1?runs-1:1));

        ofs << "N = " << N << ", Kernel = Tiled\n";
        ofs << "  checksum = 0x" << hex << chkTiled << dec << "\n";
        if(validate) ofs << "  erro_relativo_max = " << maxDiffTiled << " " << (maxDiffTiled < 1e-8 ? "[OK]" : "[ERRO]") << "\n";
        else ofs << "  erro_relativo_max = N/A (sem --validate)\n";
        ofs << "  tempos: "; for(auto t:t_tiled) ofs << t << ", "; ofs << "\n";
        ofs << "  media: " << meanT << " s, desvio: " << sdT << " s\n";
        ofs << "  Speedup vs Naive: " << (meanN / meanT) << "x\n\n";

        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    }

    cout << "Finalizado. Resultados em " << outFile << endl;
    return 0;
}