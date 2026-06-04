/* qwen_tts_server.h - Minimal HTTP server for Qwen3-TTS */
#ifndef QWEN_TTS_SERVER_H
#define QWEN_TTS_SERVER_H

#include "qwen_tts.h"

/* Start HTTP server. Blocks until killed. Returns 0 on clean shutdown, -1 on error. */
int qwen_tts_serve(qwen_tts_ctx_t *ctx, int port);

/* Like qwen_tts_serve, but with n_workers concurrent synthesis workers.
 *   n_workers <= 1 : single-threaded inline accept loop (original behavior).
 *   n_workers >= 2 : acceptor thread + worker pool; worker 0 uses `ctx`, the
 *                    rest are independent clones (qwen_tts_clone_for_worker).
 * On thread-pool backends that are NOT reentrant (pthread/Win32) synthesis is
 * serialized with an internal lock — correct but no intra-op overlap; full
 * parallelism only on the GCD backend. */
int qwen_tts_serve_ex(qwen_tts_ctx_t *ctx, int port, int n_workers);

#endif /* QWEN_TTS_SERVER_H */
