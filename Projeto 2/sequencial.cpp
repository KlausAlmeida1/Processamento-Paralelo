/*
Dupla: Adrielle e Klaus
Versão revisada para rodadas múltiplas com média/desvio e saída em TXT.
*/

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <string>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <cmath>

using namespace std;

static void imprimirMatriz(const double* M, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) cout << M[i * n + j] << "\t";
        cout << "\n";
    }
}

static void preencherMatriz(double* M, int n, mt19937_64& gen) {
    uniform_real_distribution<double> dist(0.0, 10.0);
    for (long long i = 0; i < 1LL * n * n; i++) M[i] = dist(gen);
}

static void dgemm_seq(const double* A, const double* B, double* C, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            double soma = 0.0;
            for (int k = 0; k < N; ++k) {
                soma += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = soma;
        }
    }
}

static uint64_t checksum64(const double* C, int N) {
    // checksum simples para facilitar conferência entre execuções/versões
    const uint64_t* p = reinterpret_cast<const uint64_t*>(C);
    size_t words = (size_t)N * (size_t)N * sizeof(double) / sizeof(uint64_t);
    uint64_t h = 0x9e3779b97f4a7c15ULL;
    for (size_t i = 0; i < words; ++i) {
        uint64_t x = p[i];
        h ^= x + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    }
    return h;
}

// parse helpers
static vector<int> parse_int_list(const string& csv) {
    vector<int> out;
    string cur;
    stringstream ss(csv);
    while (getline(ss, cur, ',')) {
        if (!cur.empty()) out.push_back(stoi(cur));
    }
    return out;
}

int main(int argc, char** argv) {
    // Defaults
    vector<int> sizes = {512, 1024, 2048, 4096};
    int runs = 5;
    bool regenPerRun = false;
    string outFile = "resultados_sequencial.txt";
    uint64_t seed = 42; // semente fixa para reprodutibilidade

    // CLI
    for (int i = 1; i < argc; ++i) {
        string a = argv[i];
        if (a == "--runs" && i + 1 < argc) runs = stoi(argv[++i]);
        else if (a == "--sizes" && i + 1 < argc) sizes = parse_int_list(argv[++i]);
        else if (a == "--regen-per-run") regenPerRun = true;
        else if (a == "--out" && i + 1 < argc) outFile = argv[++i];
        else if (a == "--seed" && i + 1 < argc) seed = stoull(argv[++i]);
        else if (a == "--help") {
            cout << "Uso: ./sequencial [--runs N] [--sizes 512,1024,...] "
                 << "[--regen-per-run] [--out arquivo.txt] [--seed S]\n";
            return 0;
        }
    }

    ofstream ofs(outFile);
    if (!ofs) {
        cerr << "Erro ao abrir arquivo de saida: " << outFile << "\n";
        return 1;
    }

    ofs << "DGEMM Sequencial — Rodadas multiplas\n";
    ofs << "runs=" << runs << ", regenPerRun=" << (regenPerRun ? "true" : "false")
        << ", seed=" << seed << "\n\n";
    ofs << fixed << setprecision(6);

    mt19937_64 gen(seed);

    for (int N : sizes) {
        vector<double> A(N * N), B(N * N), C(N * N);
        if (!regenPerRun) {
            preencherMatriz(A.data(), N, gen);
            preencherMatriz(B.data(), N, gen);
        }

        vector<double> tempos;
        tempos.reserve(runs);
        uint64_t lastChk = 0;

        // warm-up opcional (curto) para “acordar” o binário/caches
        {
            vector<double> Cw(N * N, 0.0);
            const double* pA = nullptr; const double* pB = nullptr;
            if (!regenPerRun) { pA = A.data(); pB = B.data(); }
            else {
                vector<double> Aw(N * N), Bw(N * N);
                preencherMatriz(Aw.data(), N, gen);
                preencherMatriz(Bw.data(), N, gen);
                pA = Aw.data(); pB = Bw.data();
            }
            dgemm_seq(pA, pB, Cw.data(), N);
        }

        for (int r = 0; r < runs; ++r) {
            if (regenPerRun) {
                preencherMatriz(A.data(), N, gen);
                preencherMatriz(B.data(), N, gen);
            }

            auto t0 = chrono::high_resolution_clock::now();
            dgemm_seq(A.data(), B.data(), C.data(), N);
            auto t1 = chrono::high_resolution_clock::now();
            chrono::duration<double> dt = t1 - t0;
            tempos.push_back(dt.count());

            lastChk = checksum64(C.data(), N);
        }

        // estatísticas
        double mean = accumulate(tempos.begin(), tempos.end(), 0.0) / tempos.size();
        double var = 0.0;
        for (double x : tempos) var += (x - mean) * (x - mean);
        var /= (tempos.size() > 1 ? (tempos.size() - 1) : 1);
        double sd = sqrt(var);

        ofs << "N = " << N << "\n";
        ofs << "  checksum(C) = 0x" << hex << lastChk << dec << "\n";
        ofs << "  tempos (s): ";
        for (size_t i = 0; i < tempos.size(); ++i) {
            ofs << tempos[i] << (i + 1 < tempos.size() ? ", " : "\n");
        }
        ofs << "  media (s): " << mean << "\n";
        ofs << "  desvio_padrao (s): " << sd << "\n\n";
    }

    cout << "OK: resultados escritos em " << outFile << "\n";
    return 0;
}
