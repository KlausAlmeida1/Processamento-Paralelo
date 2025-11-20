// Klaus e Adrielle
// Compilar: mpicxx -O3 -march=native -DNDEBUG -std=c++17 dgemm_mpi_append.cpp -o dgemm_mpi
// Exemplo:  mpirun -np 2 ./dgemm_mpi --runs 7 --sizes 512,1024 --out resultados_mpi.txt --append
#include <mpi.h>
#include <vector>
#include <random>
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <cmath>
#include <cstdint>
#include <sys/stat.h>

using namespace std;

static bool file_exists(const string& path){
    struct stat st{};
    return stat(path.c_str(), &st) == 0;
}

static void preencherMatriz(double* M, int n, mt19937_64& gen) {
    uniform_real_distribution<double> dist(0.0, 10.0);
    long long NN = 1LL * n * n;
    for (long long i = 0; i < NN; ++i) M[i] = dist(gen);
}

static void transpose_to(const double* B, double* Bt, int N) {
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            Bt[j * N + i] = B[i * N + j];
}

static void dgemm_seq_bt_rows(const double* Arows, const double* Bt, double* Crows, int rows, int N) {
    for (int i = 0; i < rows; ++i) {
        const double* Ai = &Arows[i * N];
        for (int j = 0; j < N; ++j) {
            const double* Btj = &Bt[j * N];
            double soma = 0.0;
            for (int k = 0; k < N; ++k) soma += Ai[k] * Btj[k];
            Crows[i * N + j] = soma;
        }
    }
}

static uint64_t checksum64(const double* C, long long elems) {
    const uint64_t* p = reinterpret_cast<const uint64_t*>(C);
    size_t words = (size_t)elems * sizeof(double) / sizeof(uint64_t);
    uint64_t h = 0x9e3779b97f4a7c15ULL;
    for (size_t i = 0; i < words; ++i) {
        uint64_t x = p[i];
        h ^= x + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    }
    return h;
}

static vector<int> parse_int_list(const string& csv) {
    vector<int> out;
    string cur; stringstream ss(csv);
    while (getline(ss, cur, ',')) if (!cur.empty()) out.push_back(stoi(cur));
    return out;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    vector<int> sizes = {512,1024,2048,4096};
    int runs = 5;
    bool regenPerRun = false;
    bool validate = false;
    bool append_mode = false;
    string outFile = "resultados_mpi.txt";
    uint64_t seed = 42;

    for (int i = 1; i < argc; ++i) {
        string a = argv[i];
        if (a == "--runs" && i+1 < argc) runs = stoi(argv[++i]);
        else if (a == "--sizes" && i+1 < argc) sizes = parse_int_list(argv[++i]);
        else if (a == "--regen-per-run") regenPerRun = true;
        else if (a == "--validate") validate = true;
        else if (a == "--append") append_mode = true;
        else if (a == "--out" && i+1 < argc) outFile = argv[++i];
        else if (a == "--seed" && i+1 < argc) seed = stoull(argv[++i]);
        else if (a == "--help") {
            if (rank == 0) {
                cout << "Uso: mpirun -np P ./dgemm_mpi [--runs N] [--sizes 512,1024,...] "
                        "[--regen-per-run] [--validate] [--out arquivo.txt] [--append] [--seed S]\n";
            }
            MPI_Finalize();
            return 0;
        }
    }

    ofstream ofs;
    if (rank == 0) {
        ios::openmode mode = ios::out;
        bool exists = file_exists(outFile);
        if (append_mode) mode |= ios::app; // sempre anexa se --append
        ofs.open(outFile, mode);
        if (!ofs) { cerr << "Erro ao abrir " << outFile << "\n"; MPI_Abort(MPI_COMM_WORLD, 2); }

        // Cabeçalho só escreve se não existe ou não estamos no modo append
        if (!append_mode || !exists) {
            ofs << "DGEMM MPI — rodadas multiplas\n";
        }
        ofs << "runs=" << runs << ", regenPerRun=" << (regenPerRun ? "true" : "false")
            << ", seed=" << seed << ", procs=" << nprocs
            << ", validate=" << (validate ? "true" : "false") << "\n\n";
        ofs << fixed << setprecision(6);
    }

    mt19937_64 gen(seed);

    for (int N : sizes) {
        int base = N / nprocs, rem = N % nprocs;
        vector<int> rows(nprocs), countsA(nprocs), displsA(nprocs), countsC(nprocs), displsC(nprocs);
        int offset = 0;
        for (int p = 0; p < nprocs; ++p) {
            rows[p] = base + (p < rem ? 1 : 0);
            countsA[p] = rows[p] * N;
            displsA[p] = offset;
            countsC[p] = rows[p] * N;
            displsC[p] = offset;
            offset += rows[p] * N;
        }
        int myRows = rows[rank];

        vector<double> A_sub((long long)myRows * N);
        vector<double> C_sub((long long)myRows * N);
        vector<double> B_full(1LL * N * N);
        vector<double> Bt_full(1LL * N * N);

        vector<double> A_full, C_full;
        if (rank == 0) {
            A_full.resize(1LL * N * N);
            C_full.resize(1LL * N * N);
        }

        if (rank == 0) {
            preencherMatriz(B_full.data(), N, gen);
            preencherMatriz(A_full.data(), N, gen);
        }
        MPI_Scatterv(rank==0? A_full.data():nullptr, countsA.data(), displsA.data(),
                     MPI_DOUBLE, A_sub.data(), myRows * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(B_full.data(), N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        transpose_to(B_full.data(), Bt_full.data(), N);
        dgemm_seq_bt_rows(A_sub.data(), Bt_full.data(), C_sub.data(), myRows, N);
        MPI_Gatherv(C_sub.data(), myRows * N, MPI_DOUBLE,
                    rank==0? C_full.data():nullptr, countsC.data(), displsC.data(),
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);

        vector<double> tempos; tempos.reserve(runs);
        uint64_t lastChk = 0;

        for (int r = 0; r < runs; ++r) {
            if (rank == 0) {
                if (regenPerRun || r == 0) {
                    preencherMatriz(B_full.data(), N, gen);
                    preencherMatriz(A_full.data(), N, gen);
                }
            }
            MPI_Barrier(MPI_COMM_WORLD);
            double t0 = MPI_Wtime();

            MPI_Scatterv(rank==0? A_full.data():nullptr, countsA.data(), displsA.data(),
                         MPI_DOUBLE, A_sub.data(), myRows * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast(B_full.data(), N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            transpose_to(B_full.data(), Bt_full.data(), N);
            dgemm_seq_bt_rows(A_sub.data(), Bt_full.data(), C_sub.data(), myRows, N);
            MPI_Gatherv(C_sub.data(), myRows * N, MPI_DOUBLE,
                        rank==0? C_full.data():nullptr, countsC.data(), displsC.data(),
                        MPI_DOUBLE, 0, MPI_COMM_WORLD);

            double t1 = MPI_Wtime();
            double dt = t1 - t0;

            if (rank == 0) {
                tempos.push_back(dt);
                lastChk = checksum64(C_full.data(), 1LL * N * N);
                if (validate) {
                    vector<double> Bt(1LL * N * N), C_seq(1LL * N * N);
                    transpose_to(B_full.data(), Bt.data(), N);
                    dgemm_seq_bt_rows(A_full.data(), Bt.data(), C_seq.data(), N, N);
                    const double eps = 1e-12;
                    double max_rel = 0.0;
                    for (long long i = 0; i < 1LL * N * N; ++i) {
                        double denom = fabs(C_seq[i]) + eps;
                        double rel = fabs(C_full[i] - C_seq[i]) / denom;
                        if (rel > max_rel) max_rel = rel;
                    }
                    ofs << "N = " << N << ", procs = " << nprocs << ", rodada = " << (r+1) << "\n";
                    ofs << "  validação (dif_rel_max): " << setprecision(12) << max_rel << setprecision(6)
                        << (max_rel <= 1e-9 ? " [OK]\n" : " [ALERTA]\n");
                    ofs.flush();
                }
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }

        if (rank == 0) {
            double mean = accumulate(tempos.begin(), tempos.end(), 0.0) / tempos.size();
            double var = 0.0;
            for (double x : tempos) var += (x - mean) * (x - mean);
            var /= (tempos.size() > 1 ? (tempos.size() - 1) : 1);
            double sd = sqrt(var);

            ofs << "N = " << N << ", procs = " << nprocs << "\n";
            ofs << "  checksum(C) = 0x" << hex << lastChk << dec << "\n";
            ofs << "  tempos (s): ";
            for (size_t i = 0; i < tempos.size(); ++i) {
                ofs << tempos[i] << (i + 1 < tempos.size() ? ", " : "\n");
            }
            ofs << "  media (s): " << mean << "\n";
            ofs << "  desvio_padrao (s): " << sd << "\n\n";
            ofs.flush();
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    if (rank == 0) {
        ofs << "# Fim\n";
        ofs.close();
        cout << "OK: resultados escritos em " << outFile << (append_mode? " (append)" : "") << "\n";
    }

    MPI_Finalize();
    return 0;
}
