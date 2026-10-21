"""Artifact transformation: construct Splash kernels outside JAX tracing."""


def fix_splash_construction(source):
    start = source.index("\ndef install_splash_attention():")
    end = source.index("\ndef hidden_only(", start)
    section = source[start:end]
    section = section.replace('    @functools.lru_cache(maxsize=8)\n    def attention_kernel(length, heads, local):',
                              '    def build_attention_kernel(length, heads, local):', 1)
    section = section.replace('    class SplashAttention(native_attention):', '''    # Build both fixed-shape kernels in ordinary Python, before any jit/vmap/grad.
    # Building them inside a cached factory during tracing saves escaped tracers.
    prepared_kernels = {
        False: build_attention_kernel(1024, 4, False),
        True: build_attention_kernel(1024, 4, True),
    }

    def attention_kernel(length, heads, local):
        if length != 1024 or heads != 4:
            raise ValueError("V3.3 requires the prebuilt 1024-position, 4-head Gemma kernels.")
        return prepared_kernels[bool(local)]

    class SplashAttention(native_attention):''', 1)
    return source[:start] + section + source[end:]
