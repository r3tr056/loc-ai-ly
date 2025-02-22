# loc-ai-ly: Local AI for Everyone

**Making Large Language Model Inference Accessible on Consumer Hardware**

![loc-ai-ly Logo Placeholder](docs/loc-ai-ly-logo.png) <!-- Replace with your project logo if you have one -->

**Project Goal:**

`loc-ai-ly` is a personal project dedicated to democratizing access to Large Language Models (LLMs).  The primary goal is to create a high-performance, efficient, and user-friendly inference engine that allows individuals to run powerful LLMs on consumer-grade hardware, including low-cost GPUs and even CPUs. We believe everyone should be able to experience the power of AI locally, without requiring expensive server infrastructure.

**Key Features:**

`loc-ai-ly` is built from the ground up with efficiency and accessibility in mind. Key features include:

*   **Optimized for Consumer Hardware:** Designed to run smoothly on readily available GPUs (like NVIDIA GeForce series) and CPUs, making LLM inference accessible to a wider audience.
*   **GGUF Model Format Support:**  Loads models in the efficient and widely adopted GGUF format, allowing you to utilize a vast ecosystem of pre-quantized models.
*   **Mixed Precision Inference:** Leverages mixed precision (FP16, BF16, FP32) to balance performance and accuracy, taking advantage of Tensor Cores on NVIDIA GPUs where available.
*   **INT8 Weight Quantization:** Implements INT8 weight quantization to significantly reduce memory footprint and accelerate inference, especially on hardware with INT8 acceleration capabilities.
*   **Weight-Only Quantization:** Offers weight-only quantization options for further memory reduction with minimal accuracy impact.
*   **Highly Optimized CUDA Kernels:** Hand-crafted and continuously refined CUDA kernels for core operations (RMSNorm, RoPE, Attention, SwiGLU, Softmax) to maximize GPU utilization and memory bandwidth.
*   **Kernel Fusion:** Aggressively fuses multiple operations into single CUDA kernels (e.g., RMSNorm + GEMM, Attention Score + Softmax + Value Weighting, SwiGLU + GEMM) to reduce kernel launch overhead and improve data locality.
*   **CUDA Graph Capture:** Utilizes CUDA Graphs to pre-compile kernel execution sequences, minimizing CPU overhead and further accelerating repetitive inference workloads.
*   **SentencePiece Tokenizer Integration:** Includes a fast and accurate SentencePiece tokenizer for efficient text processing, compatible with Llama family models.
*   **OpenAI-Compatible API Server:** Provides an OpenAI-compatible REST API server, allowing seamless integration with existing tools and libraries that utilize the OpenAI API standard (like the `openai` Python library).
*   **Model Manager & Store:** Features a basic model repository to manage and load multiple models from a local model store, enabling easy switching between different LLMs.
*   **BF16 Support:** Full support for the BF16 (BFloat16) data type for increased numerical stability and performance on compatible hardware.
*   **Runtime Quantization Switching:** Allows users to switch between different quantization levels at runtime without reloading the model, providing flexibility for various performance and accuracy trade-offs.

**Getting Started:**

**Prerequisites:**

*   **NVIDIA GPU with CUDA:** For GPU acceleration, you'll need an NVIDIA GPU and the CUDA Toolkit installed. Ensure your CUDA version is compatible with the code.
*   **SentencePiece Library:** Install the SentencePiece library. You can typically install it using your system's package manager (e.g., `apt install libsentencepiece-dev` on Debian/Ubuntu, `brew install sentencepiece` on macOS) or build it from source.
*   **cpp-httplib:**  Download or clone the `cpp-httplib` library and place the `httplib.h` header file in the `include/` directory of the project.
*   **nlohmann/json:** Download or clone the `nlohmann/json` library and ensure the `nlohmann` directory is in your `include/` directory, or install it via a package manager (e.g., `apt install libnlohmann-json-dev`).
*   **CMake:** CMake (version 3.15 or higher) is required for building the project.

**Build Instructions:**

1.  **Clone the repository:**
    ```bash
    git clone [repository URL]
    cd loc-ai-ly
    mkdir build
    cd build
    ```

2.  **Configure CMake:**
    ```bash
    cmake ..
    ```
    *   If CMake fails to find SentencePiece, httplib, or nlohmann/json, you may need to adjust the `CMakeLists.txt` file to point to their correct installation paths.

3.  **Build the project:**
    ```bash
    make
    ```

**Running the API Server:**

1.  **Place GGUF Models:** Download GGUF format model files (e.g., Llama 2, Llama 3 models in GGUF format) and place them in the `models/` directory within the project root. Ensure you also have the corresponding SentencePiece tokenizer model (`.tokenizer.model` file) in the same directory, named like `<model_name>.tokenizer.model`.

2.  **Run the executable:**
    ```bash
    ./llama_api_server
    ```
    The server will start listening on `http://localhost:8080`.

**Using the Python Test Client:**

1.  **Install OpenAI Python Library:**
    ```bash
    pip install openai requests
    ```

2.  **Run the test client:**
    ```bash
    python src/test_client.py
    ```
    This script will send a chat completion request to the API server and print the response. You can modify the `test_client.py` to experiment with different prompts and models.

**Model Compatibility:**

`loc-ai-ly` is currently designed and tested for:

*   **Model Architectures:** Llama 2 and Llama 3 family of models (and structurally similar architectures).
*   **Model Format:** GGUF format model files.
*   **Tokenizer:** SentencePiece tokenizers (typically used with Llama models).

While designed for Llama family models, the architecture is intended to be extensible to support other model types in the future.

**Performance:**

`loc-ai-ly` prioritizes performance on consumer hardware through:

*   **Reduced Memory Footprint:** Quantization techniques (INT8) significantly reduce memory usage, allowing larger models to fit on lower-memory GPUs and CPUs.
*   **Accelerated Computation:** Mixed precision and optimized CUDA kernels leverage GPU hardware acceleration (like Tensor Cores) for faster inference.
*   **Reduced Overhead:** Kernel fusion and CUDA Graph Capture minimize kernel launch overhead and improve overall inference speed.

Performance will vary depending on the specific model, quantization level, hardware, and input sequence lengths. Benchmarking and tuning are essential for achieving optimal performance on your target hardware.

**Future Work & Roadmap:**

The project is under active development, and future enhancements are planned:

*   **INT4 Quantization:** Implementing INT4 quantization for even greater memory reduction and potential speedups.
*   **More Kernel Fusion:** Expanding kernel fusion to other operation sequences within the Transformer architecture for further performance gains.
*   **Algorithm Selection Tuning:** Implementing automated and user-configurable GEMM algorithm selection for `cublasGemmEx` to optimize performance for different data types and matrix sizes.
*   **Broader Model Support:**  Extending support to other model architectures beyond the Llama family (e.g., Mistral, potentially GPT-like models).
*   **Advanced Sampling Methods:** Implementing more advanced sampling strategies beyond greedy decoding (e.g., beam search, nucleus sampling).
*   **Streaming Responses:**  Implementing streaming API responses for lower perceived latency.
*   **Improved Documentation & Examples:** Expanding documentation and providing more comprehensive examples for different use cases.
*   **Python SDK:** Developing a more feature-rich and user-friendly Python SDK for interacting with the inference engine.

**Contributing:**

Contributions are welcome! If you are interested in contributing to `loc-ai-ly`, please feel free to:

*   Report issues and feature requests.
*   Submit pull requests with bug fixes, performance improvements, or new features.
*   Help improve documentation.

Please follow coding style conventions and provide clear commit messages and pull request descriptions.

**License:**

This project is licensed under the [MIT License](LICENSE).

**Author:**

[Ankur Debnath/@r3tr056] ([dangerankur56@gmail.com/r3tr056])

---

**Disclaimer:**

`loc-ai-ly` is a personal project under development and is provided "as is" without warranty of any kind. It is not intended for critical production environments in its current state. Use at your own risk.