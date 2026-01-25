/*
 * GGML Infrastructure Test
 * 
 * Tests GGUF loading for Qwen3 and FLUX models.
 * Build: make test-ggml
 */

#include "flux_ggml.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    const char *qwen3_path = "models/unsloth/Qwen3-4B-GGUF/Qwen3-4B-Q8_0.gguf";
    const char *flux_path = "models/unsloth/FLUX.2-klein-4B-GGUF/flux-2-klein-4b-Q8_0.gguf";
    
    if (argc > 1) qwen3_path = argv[1];
    if (argc > 2) flux_path = argv[2];
    
    printf("=== GGML Infrastructure Test ===\n\n");
    
    /* Initialize backend */
    printf("1. Initializing backend...\n");
    flux_ggml_backend_t *backend = flux_ggml_backend_init();
    if (!backend) {
        fprintf(stderr, "FATAL: Backend init failed\n");
        return 1;
    }
    printf("\n");
    
    /* Load Qwen3 model */
    printf("2. Loading Qwen3 text encoder...\n");
    flux_ggml_model_t *qwen3 = flux_ggml_model_load(qwen3_path);
    if (!qwen3) {
        fprintf(stderr, "FATAL: Qwen3 load failed\n");
        return 1;
    }
    printf("\n");
    
    /* Print Qwen3 tensors */
    printf("3. Qwen3 tensor structure:\n");
    flux_ggml_model_print_tensors(qwen3, 15);
    printf("\n");
    
    /* Check Qwen3 metadata */
    printf("4. Qwen3 metadata:\n");
    const char *arch = flux_ggml_model_get_str(qwen3, "general.architecture");
    uint32_t n_layers = flux_ggml_model_get_u32(qwen3, "qwen3.block_count", 0);
    uint32_t n_embd = flux_ggml_model_get_u32(qwen3, "qwen3.embedding_length", 0);
    uint32_t n_head = flux_ggml_model_get_u32(qwen3, "qwen3.attention.head_count", 0);
    printf("  Architecture: %s\n", arch ? arch : "(not found)");
    printf("  Layers: %u\n", n_layers);
    printf("  Embedding dim: %u\n", n_embd);
    printf("  Attention heads: %u\n", n_head);
    printf("\n");
    
    /* Test tensor lookup */
    printf("5. Testing tensor lookup:\n");
    int64_t idx = flux_ggml_model_find_tensor(qwen3, "token_embd.weight");
    printf("  token_embd.weight: index=%lld\n", (long long)idx);
    idx = flux_ggml_model_find_tensor(qwen3, "blk.0.attn_q.weight");
    printf("  blk.0.attn_q.weight: index=%lld\n", (long long)idx);
    idx = flux_ggml_model_find_tensor(qwen3, "nonexistent");
    printf("  nonexistent: index=%lld (expected -1)\n", (long long)idx);
    printf("\n");
    
    /* Load FLUX model */
    printf("6. Loading FLUX transformer...\n");
    flux_ggml_model_t *flux = flux_ggml_model_load(flux_path);
    if (!flux) {
        fprintf(stderr, "FATAL: FLUX load failed\n");
        return 1;
    }
    printf("\n");
    
    /* Print FLUX tensors */
    printf("7. FLUX tensor structure:\n");
    flux_ggml_model_print_tensors(flux, 15);
    printf("\n");
    
    /* Cleanup */
    printf("8. Cleanup...\n");
    flux_ggml_model_free(qwen3);
    flux_ggml_model_free(flux);
    flux_ggml_backend_free();
    
    printf("\n=== All tests passed! ===\n");
    return 0;
}
