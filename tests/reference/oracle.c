/* Test-only process boundary around the unmodified LibAtrac9 sources in this
 * directory. The protocol is described in README.md. */
#include "decinit.h"
#include "decoder.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#include <windows.h>
#else
#include <time.h>
#endif

static int read_exact(void *p, size_t n) { return fread(p, 1, n, stdin) == n; }
static void put32(uint32_t n) { fwrite(&n, 4, 1, stdout); }

#ifdef _WIN32
static uint64_t elapsed_ns(LARGE_INTEGER start) {
    LARGE_INTEGER end, frequency;
    QueryPerformanceCounter(&end);
    QueryPerformanceFrequency(&frequency);
    return (uint64_t)((end.QuadPart - start.QuadPart) * 1000000000.0 / frequency.QuadPart);
}
#define CLOCK_START(start) LARGE_INTEGER start; QueryPerformanceCounter(&start)
#else
static uint64_t elapsed_ns(struct timespec start) {
    struct timespec end;
    clock_gettime(CLOCK_MONOTONIC, &end);
    return (uint64_t)(end.tv_sec - start.tv_sec) * 1000000000u +
        (uint64_t)end.tv_nsec - (uint64_t)start.tv_nsec;
}
#define CLOCK_START(start) struct timespec start; clock_gettime(CLOCK_MONOTONIC, &start)
#endif

static int timed_decode(Atrac9Handle *handle, unsigned char **frames,
        uint32_t count, uint32_t repeats, uint64_t *ns) {
    short pcm[MAX_FRAME_SAMPLES * MAX_CHANNEL_COUNT];
    volatile short checksum = 0;
    CLOCK_START(start);
    for (uint32_t repeat = 0; repeat < repeats; ++repeat) {
        for (uint32_t i = 0; i < count; ++i) {
            int used = 0;
            int status = Decode(handle, frames[i], (unsigned char *)pcm, &used);
            if (status) return 8;
            checksum ^= pcm[0];
        }
    }
    *ns = elapsed_ns(start);
    (void)checksum;
    return 0;
}

/* Request: magic, config[4], frame_count:u32, repeats:u32, then length+data
 * per input. AT9O with repeats=0 decodes each input as one frame and with
 * repeats=UINT32_MAX as one superframe, emitting status, bytes used, sample
 * count, i16 samples and an f32 view of the internal f64 PCM per frame. Other
 * repeat counts return nanoseconds:u64 for the in-process decode loop. AT9B
 * starts a timing worker that answers repeat counts until zero or EOF. */
int main(void) {
    unsigned char magic[4], config[4];
    uint32_t count, repeats;
    unsigned char **frames;
    Atrac9Handle *handle;
    short pcm[MAX_FRAME_SAMPLES * MAX_CHANNEL_COUNT];
    int status, timing_worker;
#ifdef _WIN32
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
#endif
    if (!read_exact(magic, 4) ||
        (memcmp(magic, "AT9O", 4) && memcmp(magic, "AT9B", 4)) ||
        !read_exact(config, 4) || !read_exact(&count, 4) ||
        !read_exact(&repeats, 4) || !count || count > 65536) return 2;
    timing_worker = !memcmp(magic, "AT9B", 4);
    /* Reject configs for which upstream indexes its six-entry table OOB. */
    if ((config[1] >> 1 & 7) >= 6) return 3;
    frames = calloc(count, sizeof(*frames));
    handle = calloc(1, sizeof(*handle));
    if (!frames || !handle) return 4;
    for (uint32_t i = 0; i < count; ++i) {
        uint32_t length;
        if (!read_exact(&length, 4) || !length || length > 65536) return 5;
        /* Upstream bit reader peeks beyond the last code; zero guard bytes. */
        frames[i] = calloc((size_t)length + 16, 1);
        if (!frames[i] || !read_exact(frames[i], length)) return 6;
    }
    status = InitDecoder(handle, config, 16);
    put32((uint32_t)status);
    if (timing_worker && fflush(stdout)) return 10;
    if (status) return 7;
    if (timing_worker) {
        while (read_exact(&repeats, 4) && repeats) {
            uint64_t ns;
            /* Match Rust's fresh decoder construction outside each sample's
             * timer while retaining this worker and immutable encoded inputs. */
            memset(handle, 0, sizeof(*handle));
            if (InitDecoder(handle, config, 16)) return 7;
            status = timed_decode(handle, frames, count, repeats, &ns);
            if (status) return status;
            if (fwrite(&ns, 8, 1, stdout) != 1 || fflush(stdout)) return 10;
        }
    } else if (!repeats || repeats == UINT32_MAX) {
        for (uint32_t i = 0; i < count; ++i) {
            int offset = 0;
            int frame_count = repeats ? handle->Config.FramesPerSuperframe : 1;
            for (int frame = 0; frame < frame_count; ++frame) {
                int used = 0;
                status = Decode(handle, frames[i] + offset, (unsigned char *)pcm, &used);
                put32((uint32_t)status);
                put32((uint32_t)used);
                uint32_t samples = status ? 0 : handle->Config.FrameSamples * handle->Config.ChannelCount;
                put32(samples);
                fwrite(pcm, 2, samples, stdout);
                /* The original DSP keeps unrounded doubles internally. Export a
                 * normalized, clipped f32 view next to its i16 output. */
                if (!status) {
                    for (int sample = 0; sample < handle->Config.FrameSamples; ++sample) {
                        for (int channel = 0; channel < handle->Config.ChannelCount; ++channel) {
                            double value = handle->Frame.Channels[channel]->Pcm[sample] / 32768.0;
                            float normalized = (float)(value < -1 ? -1 : value > 1 ? 1 : value);
                            fwrite(&normalized, 4, 1, stdout);
                        }
                    }
                }
                offset += used;
            }
        }
    } else {
        uint64_t ns;
        status = timed_decode(handle, frames, count, repeats, &ns);
        if (status) return status;
        fwrite(&ns, 8, 1, stdout);
    }
    for (uint32_t i = 0; i < count; ++i) free(frames[i]);
    free(frames);
    free(handle);
    return ferror(stdout) ? 10 : 0;
}
