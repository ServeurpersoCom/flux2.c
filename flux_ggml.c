/*
 * FLUX GGML Backend Infrastructure
 * 
 * Unified backend using GGML for all compute operations.
 * Loads GGUF models and manages compute graphs.
 * 
 * Phase 2.1: Infrastructure - GGUF loading, backend init
 */

#include "flux_ggml.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ========================================================================
 * Backend Management
 * ======================================================================== */

struct flux_ggml_backend {
    ggml_backend_t backend;           /* Primary compute backend */
    ggml_backend_buffer_type_t buft;  /* Buffer type for allocations */
    char name[64];                    /* Backend name for logging */
};

static flux_ggml_backend_t *g_backend = NULL;

flux_ggml_backend_t *flux_ggml_backend_init(void) {
    if (g_backend) {
        return g_backend;  /* Already initialized */
    }
    
    g_backend = calloc(1, sizeof(flux_ggml_backend_t));
    if (!g_backend) {
        fprintf(stderr, "GGML: Failed to allocate backend context\n");
        return NULL;
    }
    
    /* Initialize CPU backend with BLAS support */
    g_backend->backend = ggml_backend_cpu_init();
    if (!g_backend->backend) {
        fprintf(stderr, "GGML: Failed to initialize CPU backend\n");
        free(g_backend);
        g_backend = NULL;
        return NULL;
    }
    
    g_backend->buft = ggml_backend_cpu_buffer_type();
    snprintf(g_backend->name, sizeof(g_backend->name), "CPU (BLAS)");
    
    printf("GGML: Initialized %s backend\n", g_backend->name);
    
    return g_backend;
}

void flux_ggml_backend_free(void) {
    if (g_backend) {
        if (g_backend->backend) {
            ggml_backend_free(g_backend->backend);
        }
        free(g_backend);
        g_backend = NULL;
    }
}

ggml_backend_t flux_ggml_get_backend(void) {
    return g_backend ? g_backend->backend : NULL;
}

/* ========================================================================
 * GGUF Model Loading
 * ======================================================================== */

struct flux_ggml_model {
    char path[512];                   /* Model file path */
    struct gguf_context *gguf_ctx;    /* GGUF metadata context */
    struct ggml_context *ggml_ctx;    /* Tensor context */
    ggml_backend_buffer_t buffer;     /* Backend buffer for weights */
    int64_t n_tensors;                /* Number of tensors */
    size_t total_size;                /* Total model size in bytes */
};

flux_ggml_model_t *flux_ggml_model_load(const char *path) {
    if (!path) {
        fprintf(stderr, "GGML: NULL model path\n");
        return NULL;
    }
    
    flux_ggml_model_t *model = calloc(1, sizeof(flux_ggml_model_t));
    if (!model) {
        fprintf(stderr, "GGML: Failed to allocate model context\n");
        return NULL;
    }
    
    strncpy(model->path, path, sizeof(model->path) - 1);
    
    /* Initialize GGUF context and load metadata */
    struct gguf_init_params params = {
        .no_alloc = true,   /* Don't allocate tensor data yet */
        .ctx = &model->ggml_ctx
    };
    
    model->gguf_ctx = gguf_init_from_file(path, params);
    if (!model->gguf_ctx) {
        fprintf(stderr, "GGML: Failed to load GGUF file: %s\n", path);
        free(model);
        return NULL;
    }
    
    model->n_tensors = gguf_get_n_tensors(model->gguf_ctx);
    
    /* Calculate total buffer size needed */
    model->total_size = 0;
    for (int64_t i = 0; i < model->n_tensors; i++) {
        model->total_size += gguf_get_tensor_size(model->gguf_ctx, i);
    }
    
    printf("GGML: Loaded %s\n", path);
    printf("  Tensors: %lld\n", (long long)model->n_tensors);
    printf("  Size: %.2f GB\n", model->total_size / (1024.0 * 1024.0 * 1024.0));
    
    return model;
}

void flux_ggml_model_free(flux_ggml_model_t *model) {
    if (!model) return;
    
    if (model->buffer) {
        ggml_backend_buffer_free(model->buffer);
    }
    if (model->ggml_ctx) {
        ggml_free(model->ggml_ctx);
    }
    if (model->gguf_ctx) {
        gguf_free(model->gguf_ctx);
    }
    free(model);
}

int flux_ggml_model_load_weights(flux_ggml_model_t *model) {
    if (!model || !model->gguf_ctx || !g_backend) {
        fprintf(stderr, "GGML: Invalid model or backend not initialized\n");
        return -1;
    }
    
    /* Allocate backend buffer for all tensors */
    size_t ctx_size = ggml_tensor_overhead() * model->n_tensors;
    struct ggml_init_params ctx_params = {
        .mem_size = ctx_size,
        .mem_buffer = NULL,
        .no_alloc = true,
    };
    
    /* Free old context if exists and create new one */
    if (model->ggml_ctx) {
        ggml_free(model->ggml_ctx);
    }
    model->ggml_ctx = ggml_init(ctx_params);
    if (!model->ggml_ctx) {
        fprintf(stderr, "GGML: Failed to create tensor context\n");
        return -1;
    }
    
    /* Create tensors in context */
    for (int64_t i = 0; i < model->n_tensors; i++) {
        const char *name = gguf_get_tensor_name(model->gguf_ctx, i);
        enum ggml_type type = gguf_get_tensor_type(model->gguf_ctx, i);
        
        /* Get tensor dimensions from GGUF */
        /* Note: We need to query dimensions properly - simplified here */
        struct ggml_tensor *t = ggml_new_tensor_1d(model->ggml_ctx, type, 
            gguf_get_tensor_size(model->gguf_ctx, i) / ggml_type_size(type));
        ggml_set_name(t, name);
    }
    
    /* Allocate buffer */
    model->buffer = ggml_backend_alloc_ctx_tensors(model->ggml_ctx, g_backend->backend);
    if (!model->buffer) {
        fprintf(stderr, "GGML: Failed to allocate tensor buffer\n");
        return -1;
    }
    
    /* Load tensor data from file */
    FILE *f = fopen(model->path, "rb");
    if (!f) {
        fprintf(stderr, "GGML: Failed to open file for reading: %s\n", model->path);
        return -1;
    }
    
    size_t data_offset = gguf_get_data_offset(model->gguf_ctx);
    
    for (int64_t i = 0; i < model->n_tensors; i++) {
        const char *name = gguf_get_tensor_name(model->gguf_ctx, i);
        size_t offset = data_offset + gguf_get_tensor_offset(model->gguf_ctx, i);
        size_t size = gguf_get_tensor_size(model->gguf_ctx, i);
        
        struct ggml_tensor *t = ggml_get_tensor(model->ggml_ctx, name);
        if (!t) {
            fprintf(stderr, "GGML: Tensor not found: %s\n", name);
            continue;
        }
        
        /* Read directly into tensor data */
        fseek(f, offset, SEEK_SET);
        void *data = malloc(size);
        if (data) {
            fread(data, 1, size, f);
            ggml_backend_tensor_set(t, data, 0, size);
            free(data);
        }
    }
    
    fclose(f);
    
    printf("GGML: Loaded weights (%.2f GB)\n", 
           ggml_backend_buffer_get_size(model->buffer) / (1024.0 * 1024.0 * 1024.0));
    
    return 0;
}

/* ========================================================================
 * Model Inspection (for debugging)
 * ======================================================================== */

void flux_ggml_model_print_tensors(flux_ggml_model_t *model, int max_tensors) {
    if (!model || !model->gguf_ctx) return;
    
    int64_t n = model->n_tensors;
    if (max_tensors > 0 && max_tensors < n) {
        n = max_tensors;
    }
    
    printf("GGML: Model tensors (%lld total, showing %lld):\n", 
           (long long)model->n_tensors, (long long)n);
    
    for (int64_t i = 0; i < n; i++) {
        const char *name = gguf_get_tensor_name(model->gguf_ctx, i);
        enum ggml_type type = gguf_get_tensor_type(model->gguf_ctx, i);
        size_t size = gguf_get_tensor_size(model->gguf_ctx, i);
        
        printf("  [%3lld] %-60s  %6s  %8.2f MB\n", 
               (long long)i, name, ggml_type_name(type),
               size / (1024.0 * 1024.0));
    }
}

struct ggml_tensor *flux_ggml_model_get_tensor(flux_ggml_model_t *model, const char *name) {
    if (!model || !model->ggml_ctx || !name) return NULL;
    return ggml_get_tensor(model->ggml_ctx, name);
}

int64_t flux_ggml_model_find_tensor(flux_ggml_model_t *model, const char *name) {
    if (!model || !model->gguf_ctx || !name) return -1;
    return gguf_find_tensor(model->gguf_ctx, name);
}

/* ========================================================================
 * Metadata Access
 * ======================================================================== */

const char *flux_ggml_model_get_str(flux_ggml_model_t *model, const char *key) {
    if (!model || !model->gguf_ctx || !key) return NULL;
    int64_t key_id = gguf_find_key(model->gguf_ctx, key);
    if (key_id < 0) return NULL;
    return gguf_get_val_str(model->gguf_ctx, key_id);
}

int32_t flux_ggml_model_get_i32(flux_ggml_model_t *model, const char *key, int32_t default_val) {
    if (!model || !model->gguf_ctx || !key) return default_val;
    int64_t key_id = gguf_find_key(model->gguf_ctx, key);
    if (key_id < 0) return default_val;
    return gguf_get_val_i32(model->gguf_ctx, key_id);
}

uint32_t flux_ggml_model_get_u32(flux_ggml_model_t *model, const char *key, uint32_t default_val) {
    if (!model || !model->gguf_ctx || !key) return default_val;
    int64_t key_id = gguf_find_key(model->gguf_ctx, key);
    if (key_id < 0) return default_val;
    return gguf_get_val_u32(model->gguf_ctx, key_id);
}

float flux_ggml_model_get_f32(flux_ggml_model_t *model, const char *key, float default_val) {
    if (!model || !model->gguf_ctx || !key) return default_val;
    int64_t key_id = gguf_find_key(model->gguf_ctx, key);
    if (key_id < 0) return default_val;
    return gguf_get_val_f32(model->gguf_ctx, key_id);
}
