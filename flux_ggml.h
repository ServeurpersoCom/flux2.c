/*
 * FLUX GGML Backend Infrastructure - Header
 * 
 * Unified backend using GGML for all compute operations.
 */

#ifndef FLUX_GGML_H
#define FLUX_GGML_H

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================
 * Opaque Types
 * ======================================================================== */

typedef struct flux_ggml_backend flux_ggml_backend_t;
typedef struct flux_ggml_model flux_ggml_model_t;

/* ========================================================================
 * Backend Management
 * ======================================================================== */

/**
 * Initialize the GGML backend (CPU with BLAS).
 * Call once at startup. Thread-safe singleton.
 */
flux_ggml_backend_t *flux_ggml_backend_init(void);

/**
 * Free the GGML backend.
 * Call at shutdown.
 */
void flux_ggml_backend_free(void);

/**
 * Get the underlying ggml_backend_t handle.
 */
ggml_backend_t flux_ggml_get_backend(void);

/* ========================================================================
 * GGUF Model Loading
 * ======================================================================== */

/**
 * Load a GGUF model file (metadata only, no weights).
 * @param path Path to .gguf file
 * @return Model handle or NULL on error
 */
flux_ggml_model_t *flux_ggml_model_load(const char *path);

/**
 * Free a loaded model.
 */
void flux_ggml_model_free(flux_ggml_model_t *model);

/**
 * Load the actual tensor weights into backend memory.
 * Call after flux_ggml_model_load() and flux_ggml_backend_init().
 * @return 0 on success, -1 on error
 */
int flux_ggml_model_load_weights(flux_ggml_model_t *model);

/* ========================================================================
 * Model Inspection
 * ======================================================================== */

/**
 * Print tensor names, types, and sizes.
 * @param max_tensors Maximum tensors to print (0 = all)
 */
void flux_ggml_model_print_tensors(flux_ggml_model_t *model, int max_tensors);

/**
 * Get a tensor by name.
 * @return ggml_tensor pointer or NULL if not found
 */
struct ggml_tensor *flux_ggml_model_get_tensor(flux_ggml_model_t *model, const char *name);

/**
 * Find tensor index by name.
 * @return Tensor index or -1 if not found
 */
int64_t flux_ggml_model_find_tensor(flux_ggml_model_t *model, const char *name);

/* ========================================================================
 * Metadata Access  
 * ======================================================================== */

/**
 * Get string metadata value by key.
 * @return String value or NULL if not found
 */
const char *flux_ggml_model_get_str(flux_ggml_model_t *model, const char *key);

/**
 * Get int32 metadata value by key.
 * @return Value or default_val if not found
 */
int32_t flux_ggml_model_get_i32(flux_ggml_model_t *model, const char *key, int32_t default_val);

/**
 * Get uint32 metadata value by key.
 * @return Value or default_val if not found
 */
uint32_t flux_ggml_model_get_u32(flux_ggml_model_t *model, const char *key, uint32_t default_val);

/**
 * Get float32 metadata value by key.
 * @return Value or default_val if not found
 */
float flux_ggml_model_get_f32(flux_ggml_model_t *model, const char *key, float default_val);

#ifdef __cplusplus
}
#endif

#endif /* FLUX_GGML_H */
