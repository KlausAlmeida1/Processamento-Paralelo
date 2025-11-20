# Projeto 3: Multiplicação de Matrizes (DGEMM) com CUDA

**Disciplina:** DEC107 - Processamento Paralelo  
**Curso:** Bacharelado em Ciência da Computação (UESC)  
**Autores:** Adrielle e Klaus  

---

## 📝 Descrição do Projeto

Este repositório contém a implementação e análise de desempenho do algoritmo de Multiplicação de Matrizes (DGEMM - *Double Precision General Matrix Multiplication*) comparando diferentes paradigmas de programação paralela:

1.  **Sequencial Otimizado:** Implementação base em C++ com transposição de memória.
2.  **OpenMP (CPU Multi-core):** Paralelismo de memória compartilhada.
3.  **MPI (CPU Distribuída):** Paralelismo de memória distribuída (troca de mensagens).
4.  **CUDA (GPU):** Paralelismo massivo utilizando GPGPU.

### 🚀 Estratégia de Otimização (Nivelamento)
Para garantir uma comparação justa entre CPU e GPU, todas as implementações de CPU (Sequencial, OpenMP e MPI) utilizam a técnica de **Transposição da Matriz B** (Row-Major Order) antes do cálculo. Isso garante acesso linear à memória, maximizando o uso do Cache da CPU e permitindo vetorização automática.

A versão CUDA compara duas abordagens:
* **Naive:** Acesso direto à memória global.
* **Tiled:** Uso de memória compartilhada (*Shared Memory*) para redução de tráfego na VRAM.

---

## 🛠️ Pré-requisitos

* **Compilador C++:** `g++` (suporte a C++17 recomendado).
* **MPI:** `mpich` ou `openmpi` (comando `mpicxx`).
* **CUDA Toolkit:** `nvcc` (Versão 11.0+ recomendada).
* **Hardware:** GPU Nvidia compatível com CUDA (Testado em RTX 4070 SUPER).

---

## ⚙️ Compilação e Execução

Abaixo estão as instruções para compilar e executar cada uma das 4 versões. Recomenda-se o uso da flag `-O3` para garantir a otimização máxima do compilador.

### 1. Sequencial (Otimizado)

Versão de referência (baseline) com acesso linear à memória.

**Compilar:**
```bash
g++ -O3 sequencial.cpp -o sequencial
```

**Executar:**
```bash
./sequencial --runs 7 --sizes 512,1024,2048,4096
```

### 2. OpenMP (Paralelo)

Utiliza threads para dividir o loop externo da multiplicação.

**Compilar:**
```bash
g++ -O3 -fopenmp paralelo.cpp -o paralelo
```

**Executar:**
*Define o número de threads via argumento `--threads`.*
```bash
./paralelo --runs 7 --sizes 512,1024,2048,4096 --threads 2,4,8,12
```

### 3. MPI (Distribuído)

Divide a matriz A em faixas de linhas entre os processos.

**Compilar:**
```bash
mpicxx -O3 -march=native mpi.cpp -o dgemm_mpi
```

**Executar:**
*Exemplo rodando com 4 processos.*
```bash
mpirun -np 4 ./dgemm_mpi --runs 7 --sizes 512,1024,2048,4096 --validate
```

### 4. CUDA (GPU)

Executa os kernels *Naive* e *Tiled* sequencialmente para comparação de speedup interno.

**Compilar:**
*Nota: Ajuste `-arch=sm_89` para a arquitetura da sua GPU (sm_89 = RTX 40 Series). Para compatibilidade geral, use `-arch=sm_75`.*
```bash
nvcc -O3 -arch=sm_89 cuda.cu -o cuda
```

**Executar:**
```bash
./cuda --runs 7 --sizes 512,1024,2048,4096 --validate
```

---

## 📊 Flags Comuns

Todos os executáveis suportam os seguintes argumentos de linha de comando para padronização dos testes:

* `--runs <N>`: Número de vezes que o teste será repetido para calcular a média (padrão: 5).
* `--sizes <lista>`: Lista de dimensões N separadas por vírgula (ex: `512,1024`).
* `--seed <N>`: Semente para geração aleatória (padrão: 42, garante reprodutibilidade).
* `--validate`: Ativa a verificação de corretude numérica (CPU vs GPU/MPI) calculando o erro relativo máximo.
* `--out <arquivo>`: Define o nome do arquivo de saída (padrão: `resultados_*.txt`).

---

## 🖥️ Ambiente de Teste (Referência)

Os resultados apresentados no relatório foram obtidos no seguinte ambiente:

* **OS:** WSL 2 (Ubuntu) no Windows 11.
* **CPU:** AMD Ryzen 5 7600X 6-Core Processor.
* **GPU:** NVIDIA GeForce RTX 4070 SUPER.
* **Memória RAM:** 7723 GB DDR5 para o WSL.
* **Compilador NVCC:** Cuda compilation tools, release 13.0.