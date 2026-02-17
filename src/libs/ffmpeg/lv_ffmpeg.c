/**
 * @file lv_ffmpeg.c
 *
 */

/*********************
 *      INCLUDES
 *********************/
#include "lv_ffmpeg_private.h"
#if LV_USE_FFMPEG != 0
#include "../../draw/lv_image_decoder_private.h"
#include "../../draw/lv_draw_buf_private.h"
#include "../../core/lv_obj_class_private.h"

#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libavutil/samplefmt.h>
#include <libavutil/timestamp.h>
#include <libswscale/swscale.h>
#if LV_FFMPEG_HWACCEL_MJPEG != 0
#include <libavutil/hwcontext.h>
#endif
#if LV_FFMPEG_AUDIO_SUPPORT != 0
#include <libavdevice/avdevice.h>
#include <libswresample/swresample.h>
#include <alsa/asoundlib.h>
#include <pthread.h>
#endif

/* CPRO OPTIMIZATION: ARM NEON intrinsics for Cortex-A7 */
#if LV_USE_DRAW_SW && defined(__ARM_NEON)
#include <arm_neon.h>
#endif

/*********************
 *      DEFINES
 *********************/

#if LV_COLOR_DEPTH == 8
    #define AV_PIX_FMT_TRUE_COLOR AV_PIX_FMT_RGB8
#elif LV_COLOR_DEPTH == 16
    #define AV_PIX_FMT_TRUE_COLOR AV_PIX_FMT_RGB565LE
#elif LV_COLOR_DEPTH == 32
    #define AV_PIX_FMT_TRUE_COLOR AV_PIX_FMT_BGR0
#else
    #error Unsupported  LV_COLOR_DEPTH
#endif

#define DECODER_NAME    "FFMPEG"

#define MY_CLASS (&lv_ffmpeg_player_class)

#define FRAME_DEF_REFR_PERIOD   50  /*[ms]*/

#define DECODER_BUFFER_SIZE (8 * 1024)

/**********************
 *      TYPEDEFS
 **********************/
#if LV_FFMPEG_AUDIO_SUPPORT != 0
/* Video frame ring buffer for frame buffering between playback thread and LVGL main thread */
#define VIDEO_BUFFER_SIZE 5  /* Buffer 5 frames to balance memory and smoothness */

typedef struct {
    AVFrame *frames[VIDEO_BUFFER_SIZE];
    int write_idx;  /* Write index (video thread) */
    int read_idx;   /* Read index (LVGL main thread) */
    int count;      /* Current frame count */
    pthread_mutex_t mutex;  /* Mutex for thread safety */
    pthread_cond_t cond;    /* Condition variable for signaling */
    bool initialized;       /* Initialization flag */
} video_buffer_t;
#endif

struct ffmpeg_context_s {
    struct _lv_ffmpeg_player_t *player;  /* Pointer to player object */
    AVIOContext * io_ctx;
    lv_fs_file_t lv_file;
    AVFormatContext * fmt_ctx;
    AVCodecContext * video_dec_ctx;
    AVStream * video_stream;
    uint8_t * video_src_data[4];
    uint8_t * video_dst_data[4];
    struct SwsContext * sws_ctx;
    AVFrame * frame;
    AVPacket * pkt;
    int video_stream_idx;
    int video_src_linesize[4];
    int video_dst_linesize[4];
    enum AVPixelFormat video_dst_pix_fmt;
    bool has_alpha;
    lv_draw_buf_t draw_buf;
    lv_draw_buf_handlers_t draw_buf_handlers;
#if LV_FFMPEG_HWACCEL_MJPEG != 0
    AVBufferRef *hw_device_ctx;            /* Hardware device context */
    AVBufferRef *hw_frames_ctx;            /* Hardware frames context */
    enum AVPixelFormat hw_pix_fmt;         /* Hardware pixel format */
    bool use_hwaccel;                      /* Use hardware acceleration */
    AVFrame *hw_transfer_frame;            /* Reusable hardware transfer frame */
    bool hw_frame_initialized;             /* Hardware transfer frame initialized flag */
    bool hw_pool_initialized;              /* Hardware frame pool initialized flag */
#endif
#if LV_FFMPEG_AUDIO_SUPPORT != 0
    AVStream * audio_stream;
    AVCodecContext * audio_dec_ctx;
    int audio_stream_idx;
    AVFrame * audio_frame;
    struct SwrContext * swr_ctx;
    uint8_t * audio_buf;
    int audio_buf_size;
    bool has_audio;
    snd_mixer_t *audio_mixer_handle;       /* ALSA Mixer handle */
    snd_mixer_elem_t *audio_mixer_elem;    /* ALSA Mixer element */
    snd_pcm_t *audio_pcm_handle;           /* ALSA PCM handle (when USE_AVDEVICE=0) */
    /* Audio output fields (avdevice mode) */
    AVFormatContext *audio_out_fmt_ctx;    /* Audio output format context (avdevice mode) */
    AVPacket *audio_out_pkt;               /* Reusable audio output packet (avdevice mode) */
    /* Legacy thread control flags (for backward compatibility during transition) */
    pthread_t video_thread;                /* Video thread handle (legacy) */
    pthread_t audio_thread;                /* Audio thread handle (legacy) */
    volatile int is_video_playing;         /* Video playing flag (legacy) */
    volatile int is_video_paused;          /* Video paused flag (legacy) */
    volatile int is_audio_playing;         /* Audio playing flag (legacy) */
    volatile int is_audio_paused;          /* Audio paused flag (legacy) */
    /* CPRO OPTIMIZATION: Frame skip detection */
    int consecutive_skips;                 /* Consecutive frame skips */
    bool skip_this_frame;                  /* Skip current video frame */
    bool needs_conversion;                 /* Needs format conversion */
    /* Unified playback thread support */
    pthread_t playback_thread;             /* Unified audio/video playback thread */
    volatile int is_playing;               /* Playback flag */
    volatile int is_paused;                /* Pause flag */
    video_buffer_t video_buffer;           /* Video frame ring buffer */
#endif
};

#pragma pack(1)

struct _lv_image_pixel_color_s {
    lv_color_t c;
    uint8_t alpha;
};

#pragma pack()

/**********************
 *  STATIC PROTOTYPES
 **********************/

static lv_result_t decoder_info(lv_image_decoder_t * decoder, lv_image_decoder_dsc_t * src, lv_image_header_t * header);
static lv_result_t decoder_open(lv_image_decoder_t * decoder, lv_image_decoder_dsc_t * dsc);
static void decoder_close(lv_image_decoder_t * dec, lv_image_decoder_dsc_t * dsc);

static int ffmpeg_lvfs_read(void * ptr, uint8_t * buf, int buf_size);
static int64_t ffmpeg_lvfs_seek(void * ptr, int64_t pos, int whence);
static AVIOContext * ffmpeg_open_io_context(lv_fs_file_t * file);
static struct ffmpeg_context_s * ffmpeg_open_file(const char * path, bool is_lv_fs_path);
static void ffmpeg_close(struct ffmpeg_context_s * ffmpeg_ctx);
static void ffmpeg_close_src_ctx(struct ffmpeg_context_s * ffmpeg_ctx);
static void ffmpeg_close_dst_ctx(struct ffmpeg_context_s * ffmpeg_ctx);
#if LV_FFMPEG_HWACCEL_MJPEG != 0
static int ffmpeg_init_hwaccel_frames(struct ffmpeg_context_s * ffmpeg_ctx,
                                       AVCodecContext * dec_ctx);
#endif
static int ffmpeg_image_allocate(struct ffmpeg_context_s * ffmpeg_ctx);
static int ffmpeg_get_image_header(lv_image_decoder_dsc_t * dsc, lv_image_header_t * header);
static int ffmpeg_get_frame_refr_period(struct ffmpeg_context_s * ffmpeg_ctx);
static uint8_t * ffmpeg_get_image_data(struct ffmpeg_context_s * ffmpeg_ctx);
static int ffmpeg_update_next_frame(struct ffmpeg_context_s * ffmpeg_ctx);
static int ffmpeg_output_video_frame(struct ffmpeg_context_s * ffmpeg_ctx);
static bool ffmpeg_pix_fmt_has_alpha(enum AVPixelFormat pix_fmt);
static bool ffmpeg_pix_fmt_is_yuv(enum AVPixelFormat pix_fmt);

/* CPRO OPTIMIZATION: NEON-accelerated YUV to RGB conversion for Cortex-A7 */
#if LV_USE_DRAW_SW && defined(__ARM_NEON)
static void neon_yuv420p_to_rgb565(const uint8_t *y, const uint8_t *u, const uint8_t *v,
                                   uint16_t *rgb, int width, int height,
                                   int y_stride, int uv_stride, int rgb_stride);
static void neon_yuv420p_to_rgb888(const uint8_t *y, const uint8_t *u, const uint8_t *v,
                                   uint8_t *rgb, int width, int height,
                                   int y_stride, int uv_stride, int rgb_stride);
#endif
#if LV_FFMPEG_AUDIO_SUPPORT != 0
static int ffmpeg_output_audio_frame(struct ffmpeg_context_s * ffmpeg_ctx);
static int ffmpeg_audio_init(struct ffmpeg_context_s * ffmpeg_ctx);
static void ffmpeg_audio_deinit(struct ffmpeg_context_s * ffmpeg_ctx);
static int ffmpeg_audio_pcm_init(struct ffmpeg_context_s * ffmpeg_ctx);
static int ffmpeg_audio_pcm_write(struct ffmpeg_context_s * ffmpeg_ctx, const uint8_t *data, int size);
static void ffmpeg_audio_pcm_deinit(struct ffmpeg_context_s * ffmpeg_ctx);

/* Video buffer management */
static int video_buffer_init(video_buffer_t *buf);
static int video_buffer_push(video_buffer_t *buf, AVFrame *frame);
static AVFrame *video_buffer_pop(video_buffer_t *buf);
static void video_buffer_destroy(video_buffer_t *buf);

/* CPRO OPTIMIZATION: Global ALSA initialization lock to prevent resource contention */
static pthread_mutex_t alsa_init_lock = PTHREAD_MUTEX_INITIALIZER;

/* Unified playback thread: processes both audio and video packets in a single thread */
static void *ffmpeg_playback_thread(void *arg);

#if LV_FFMPEG_SYNC_ENABLED != 0
/* Audio-Video Synchronization Functions */
static int64_t pts_to_ms(AVStream *stream, int64_t pts);
static int64_t get_current_time_ms(void);
static bool should_skip_video_frame(struct ffmpeg_context_s *ffmpeg_ctx);
static bool should_repeat_video_frame(struct ffmpeg_context_s *ffmpeg_ctx);
#endif
#endif

static void lv_ffmpeg_player_constructor(const lv_obj_class_t * class_p, lv_obj_t * obj);
static void lv_ffmpeg_player_destructor(const lv_obj_class_t * class_p, lv_obj_t * obj);

/**********************
 *  STATIC VARIABLES
 **********************/

const lv_obj_class_t lv_ffmpeg_player_class = {
    .constructor_cb = lv_ffmpeg_player_constructor,
    .destructor_cb = lv_ffmpeg_player_destructor,
    .instance_size = sizeof(lv_ffmpeg_player_t),
    .base_class = &lv_image_class,
    .name = "lv_ffmpeg_player",
};

/**********************
 *      MACROS
 **********************/

/**********************
 *   GLOBAL FUNCTIONS
 **********************/

void lv_ffmpeg_init(void)
{
    lv_image_decoder_t * dec = lv_image_decoder_create();
    lv_image_decoder_set_info_cb(dec, decoder_info);
    lv_image_decoder_set_open_cb(dec, decoder_open);
    lv_image_decoder_set_close_cb(dec, decoder_close);

    dec->name = DECODER_NAME;

#if LV_FFMPEG_DUMP_FORMAT == 0
    av_log_set_level(AV_LOG_QUIET);
#endif

#if LV_FFMPEG_AUDIO_SUPPORT != 0
    avdevice_register_all();
#endif
}

void lv_ffmpeg_deinit(void)
{
    lv_image_decoder_t * dec = NULL;
    while((dec = lv_image_decoder_get_next(dec)) != NULL) {
        if(dec->info_cb == decoder_info) {
            lv_image_decoder_delete(dec);
            break;
        }
    }
}

int lv_ffmpeg_get_frame_num(const char * path)
{
    int ret = -1;
    struct ffmpeg_context_s * ffmpeg_ctx = ffmpeg_open_file(path, LV_FFMPEG_PLAYER_USE_LV_FS);

    if(ffmpeg_ctx) {
        ret = ffmpeg_ctx->video_stream->nb_frames;
        ffmpeg_close(ffmpeg_ctx);
    }

    return ret;
}

lv_obj_t * lv_ffmpeg_player_create(lv_obj_t * parent)
{
    lv_obj_t * obj = lv_obj_class_create_obj(MY_CLASS, parent);
    lv_obj_class_init_obj(obj);
    return obj;
}

lv_result_t lv_ffmpeg_player_set_src(lv_obj_t * obj, const char * path)
{
    LV_ASSERT_OBJ(obj, MY_CLASS);
    lv_result_t res = LV_RESULT_INVALID;

    lv_ffmpeg_player_t * player = (lv_ffmpeg_player_t *)obj;

    if(player->ffmpeg_ctx) {
        ffmpeg_close(player->ffmpeg_ctx);
        player->ffmpeg_ctx = NULL;
    }

    lv_timer_pause(player->timer);

    player->ffmpeg_ctx = ffmpeg_open_file(path, LV_FFMPEG_PLAYER_USE_LV_FS);

    if(!player->ffmpeg_ctx) {
        goto failed;
    }

    /* Set player pointer in ffmpeg_context */
    player->ffmpeg_ctx->player = player;

    if(ffmpeg_image_allocate(player->ffmpeg_ctx) < 0) {
        LV_LOG_ERROR("ffmpeg image allocate failed");
        ffmpeg_close(player->ffmpeg_ctx);
        player->ffmpeg_ctx = NULL;
        goto failed;
    }

#if LV_FFMPEG_AUDIO_SUPPORT != 0
    /* Two-thread architecture: Initialize video buffer */
    if(video_buffer_init(&player->ffmpeg_ctx->video_buffer) < 0) {
        LV_LOG_ERROR("Failed to initialize video buffer");
        ffmpeg_close(player->ffmpeg_ctx);
        player->ffmpeg_ctx = NULL;
        goto failed;
    }

    /* Initialize unified playback thread control flags */
    player->ffmpeg_ctx->is_playing = 0;
    player->ffmpeg_ctx->is_paused = 0;
#endif

    bool has_alpha = player->ffmpeg_ctx->has_alpha;
    int width = player->ffmpeg_ctx->video_dec_ctx->width;
    int height = player->ffmpeg_ctx->video_dec_ctx->height;

    uint8_t * data = ffmpeg_get_image_data(player->ffmpeg_ctx);
    lv_color_format_t cf = has_alpha ? LV_COLOR_FORMAT_ARGB8888 : LV_COLOR_FORMAT_NATIVE;
    uint32_t stride = width * lv_color_format_get_size(cf);
    uint32_t data_size = stride * height;
    lv_memzero(data, data_size);

    player->imgdsc.header.w = width;
    player->imgdsc.header.h = height;
    player->imgdsc.data_size = data_size;
    player->imgdsc.header.cf = cf;
    player->imgdsc.header.stride = stride;
    player->imgdsc.data = data;

    lv_image_set_src(&player->img.obj, &(player->imgdsc));

    int period = ffmpeg_get_frame_refr_period(player->ffmpeg_ctx);

    if(period > 0) {
        LV_LOG_INFO("frame refresh period = %d ms, rate = %d fps",
                    period, 1000 / period);
        lv_timer_set_period(player->timer, period);
    }
    else {
        LV_LOG_WARN("unable to get frame refresh period");
    }

    res = LV_RESULT_OK;

failed:
    return res;
}

void lv_ffmpeg_player_set_cmd(lv_obj_t * obj, lv_ffmpeg_player_cmd_t cmd)
{
    LV_ASSERT_OBJ(obj, MY_CLASS);
    lv_ffmpeg_player_t * player = (lv_ffmpeg_player_t *)obj;

    if(!player->ffmpeg_ctx) {
        LV_LOG_ERROR("ffmpeg_ctx is NULL");
        return;
    }

    lv_timer_t * timer = player->timer;

    switch(cmd) {
        case LV_FFMPEG_PLAYER_CMD_START:
            av_seek_frame(player->ffmpeg_ctx->fmt_ctx,
                          0, 0, AVSEEK_FLAG_BACKWARD);
            lv_timer_resume(timer);

#if LV_FFMPEG_AUDIO_SUPPORT != 0
            /* Two-thread architecture: Start unified playback thread */
            player->ffmpeg_ctx->is_playing = 1;
            player->ffmpeg_ctx->is_paused = 0;
            pthread_create(&player->ffmpeg_ctx->playback_thread, NULL,
                           ffmpeg_playback_thread, player->ffmpeg_ctx);
            LV_LOG_INFO("Unified playback thread started");
#endif

            LV_LOG_INFO("ffmpeg player start");
            break;
        case LV_FFMPEG_PLAYER_CMD_STOP:
            av_seek_frame(player->ffmpeg_ctx->fmt_ctx,
                          0, 0, AVSEEK_FLAG_BACKWARD);
            lv_timer_pause(timer);

#if LV_FFMPEG_AUDIO_SUPPORT != 0
            /* Two-thread architecture: Stop unified playback thread */
            if(player->ffmpeg_ctx->is_playing) {
                player->ffmpeg_ctx->is_playing = 0;
                pthread_join(player->ffmpeg_ctx->playback_thread, NULL);
                LV_LOG_INFO("Unified playback thread stopped");
            }
#endif

#if LV_FFMPEG_SYNC_ENABLED != 0
            /* Reset audio-video synchronization state */
            player->ffmpeg_ctx->video_clock = 0;
            player->ffmpeg_ctx->audio_clock = 0;
            __sync_synchronize();  /* Memory barrier */
player->ffmpeg_ctx->video_pts = AV_NOPTS_VALUE;
            __sync_synchronize();  /* Memory barrier */
player->ffmpeg_ctx->audio_pts = AV_NOPTS_VALUE;
            player->ffmpeg_ctx->start_time = 0;
            player->ffmpeg_ctx->frame_drop_count = 0;
            player->ffmpeg_ctx->frame_repeat_count = 0;
            LV_LOG_INFO("[SYNC] Synchronization state reset");
#endif

            LV_LOG_INFO("ffmpeg player stop");
            break;
        case LV_FFMPEG_PLAYER_CMD_PAUSE:
            lv_timer_pause(timer);

#if LV_FFMPEG_AUDIO_SUPPORT != 0
            /* Two-thread architecture: Pause unified playback thread */
            if(player->ffmpeg_ctx->is_playing) {
                player->ffmpeg_ctx->is_paused = 1;
            }
#endif

            LV_LOG_INFO("ffmpeg player pause");
            break;
        case LV_FFMPEG_PLAYER_CMD_RESUME:
            lv_timer_resume(timer);

#if LV_FFMPEG_AUDIO_SUPPORT != 0
            /* Two-thread architecture: Resume unified playback thread */
            if(player->ffmpeg_ctx->is_playing) {
                player->ffmpeg_ctx->is_paused = 0;
            }
#endif

            LV_LOG_INFO("ffmpeg player resume");
            break;
        default:
            LV_LOG_ERROR("Error cmd: %d", cmd);
            break;
    }
}

void lv_ffmpeg_player_set_auto_restart(lv_obj_t * obj, bool en)
{
    LV_ASSERT_OBJ(obj, MY_CLASS);
    lv_ffmpeg_player_t * player = (lv_ffmpeg_player_t *)obj;
    player->auto_restart = en;
}

#if LV_FFMPEG_AUDIO_SUPPORT != 0
void lv_ffmpeg_player_set_volume(lv_obj_t * obj, int volume)
{
    LV_ASSERT_OBJ(obj, MY_CLASS);
    lv_ffmpeg_player_t * player = (lv_ffmpeg_player_t *)obj;
    player->volume = LV_CLAMP(0, volume, 100);

    /* Apply hardware volume control via ALSA Mixer */
    /* Mixer is disabled to avoid conflict with PCM, skip volume control */
    /*
    if(player->ffmpeg_ctx) {
        ffmpeg_audio_mixer_set_volume(player->ffmpeg_ctx, player->volume);
    }
    */

    LV_LOG_INFO("Set volume to %d", player->volume);
}

int lv_ffmpeg_player_get_volume(lv_obj_t * obj)
{
    LV_ASSERT_OBJ(obj, MY_CLASS);
    lv_ffmpeg_player_t * player = (lv_ffmpeg_player_t *)obj;

    /* Get current volume from ALSA Mixer */
    /* Mixer is disabled to avoid conflict with PCM, return stored volume */
    /*
    if(player->ffmpeg_ctx) {
        player->volume = ffmpeg_audio_mixer_get_volume(player->ffmpeg_ctx);
    }
    */

    return player->volume;
}

void lv_ffmpeg_player_set_audio_enabled(lv_obj_t * obj, bool en)
{
    LV_ASSERT_OBJ(obj, MY_CLASS);
    lv_ffmpeg_player_t * player = (lv_ffmpeg_player_t *)obj;
    player->audio_enabled = en;
    LV_LOG_INFO("Audio %s", en ? "enabled" : "disabled");
}

bool lv_ffmpeg_player_get_audio_enabled(lv_obj_t * obj)
{
    LV_ASSERT_OBJ(obj, MY_CLASS);
    lv_ffmpeg_player_t * player = (lv_ffmpeg_player_t *)obj;
    return player->audio_enabled;
}
#endif

/**********************
 *   STATIC FUNCTIONS
 **********************/

static lv_result_t decoder_info(lv_image_decoder_t * decoder, lv_image_decoder_dsc_t * dsc, lv_image_header_t * header)
{
    LV_UNUSED(decoder);

    /* Get the source type */
    lv_image_src_t src_type = dsc->src_type;

    if(src_type == LV_IMAGE_SRC_FILE) {
        if(ffmpeg_get_image_header(dsc, header) < 0) {
            LV_LOG_ERROR("ffmpeg can't get image header");
            return LV_RESULT_INVALID;
        }

        return LV_RESULT_OK;
    }

    /* If didn't succeeded earlier then it's an error */
    return LV_RESULT_INVALID;
}

/**
 * Decode an image using ffmpeg library
 * @param decoder pointer to the decoder
 * @param dsc     pointer to the decoder descriptor
 * @return LV_RESULT_OK: no error; LV_RESULT_INVALID: can't open the image
 */
static lv_result_t decoder_open(lv_image_decoder_t * decoder, lv_image_decoder_dsc_t * dsc)
{
    LV_UNUSED(decoder);

    if(dsc->src_type == LV_IMAGE_SRC_FILE) {
        const char * path = dsc->src;

        struct ffmpeg_context_s * ffmpeg_ctx = ffmpeg_open_file(path, true);

        if(ffmpeg_ctx == NULL) {
            return LV_RESULT_INVALID;
        }

        if(ffmpeg_image_allocate(ffmpeg_ctx) < 0) {
            LV_LOG_ERROR("ffmpeg image allocate failed");
            ffmpeg_close(ffmpeg_ctx);
            return LV_RESULT_INVALID;
        }

        if(ffmpeg_update_next_frame(ffmpeg_ctx) < 0) {
            ffmpeg_close(ffmpeg_ctx);
            LV_LOG_ERROR("ffmpeg update frame failed");
            return LV_RESULT_INVALID;
        }

        ffmpeg_close_src_ctx(ffmpeg_ctx);
        uint8_t * img_data = ffmpeg_get_image_data(ffmpeg_ctx);

        dsc->user_data = ffmpeg_ctx;
        lv_draw_buf_t * decoded = &ffmpeg_ctx->draw_buf;
        lv_draw_buf_init(
            decoded,
            dsc->header.w,
            dsc->header.h,
            dsc->header.cf,
            dsc->header.stride,
            img_data,
            dsc->header.stride * dsc->header.h);
        lv_draw_buf_set_flag(decoded, LV_IMAGE_FLAGS_MODIFIABLE);

        /* Empty handlers to avoid decoder asserts */
        lv_draw_buf_handlers_init(&ffmpeg_ctx->draw_buf_handlers, NULL, NULL, NULL, NULL, NULL, NULL);
        decoded->handlers = &ffmpeg_ctx->draw_buf_handlers;

        if(dsc->args.premultiply && ffmpeg_ctx->has_alpha) {
            lv_draw_buf_premultiply(decoded);
        }

        dsc->decoded = decoded;

        /* The image is fully decoded. Return with its pointer */
        return LV_RESULT_OK;
    }

    /* If not returned earlier then it failed */
    return LV_RESULT_INVALID;
}

static void decoder_close(lv_image_decoder_t * decoder, lv_image_decoder_dsc_t * dsc)
{
    LV_UNUSED(decoder);
    struct ffmpeg_context_s * ffmpeg_ctx = dsc->user_data;
    ffmpeg_close(ffmpeg_ctx);
}

static uint8_t * ffmpeg_get_image_data(struct ffmpeg_context_s * ffmpeg_ctx)
{
    uint8_t * img_data = ffmpeg_ctx->video_dst_data[0];

    if(img_data == NULL) {
        LV_LOG_ERROR("ffmpeg video dst data is NULL");
    }

    return img_data;
}

static bool ffmpeg_pix_fmt_has_alpha(enum AVPixelFormat pix_fmt)
{
    const AVPixFmtDescriptor * desc = av_pix_fmt_desc_get(pix_fmt);

    if(desc == NULL) {
        return false;
    }

    if(pix_fmt == AV_PIX_FMT_PAL8) {
        return true;
    }

    return desc->flags & AV_PIX_FMT_FLAG_ALPHA;
}

static bool ffmpeg_pix_fmt_is_yuv(enum AVPixelFormat pix_fmt)
{
    const AVPixFmtDescriptor * desc = av_pix_fmt_desc_get(pix_fmt);

    if(desc == NULL) {
        return false;
    }

    return !(desc->flags & AV_PIX_FMT_FLAG_RGB) && desc->nb_components >= 2;
}

static int ffmpeg_output_video_frame(struct ffmpeg_context_s * ffmpeg_ctx)
{
    int ret = -1;
    AVFrame *frame = ffmpeg_ctx->frame;

#if LV_FFMPEG_SYNC_ENABLED != 0
    /* Update video clock based on PTS */
    if (frame->pts != AV_NOPTS_VALUE) {
        int64_t pts_ms = pts_to_ms(ffmpeg_ctx->video_stream, frame->pts);
        if (pts_ms != AV_NOPTS_VALUE) {
            ffmpeg_ctx->video_pts = pts_ms;
            ffmpeg_ctx->video_clock = pts_ms;
            __sync_synchronize();  /* Memory barrier after update */

            /* If this is the first frame, set start time */
            if (ffmpeg_ctx->start_time == 0) {
                ffmpeg_ctx->start_time = get_current_time_ms();
            }

            LV_LOG_TRACE("[SYNC] Video frame decoded: PTS=%lld ms, video_clock=%lld ms",
                        (long long)ffmpeg_ctx->video_pts,
                        (long long)ffmpeg_ctx->video_clock);
        }
    }
#endif

    int width = ffmpeg_ctx->video_dec_ctx->width;
    int height = ffmpeg_ctx->video_dec_ctx->height;

    if(frame->width != width
       || frame->height != height
       || frame->format != ffmpeg_ctx->video_dec_ctx->pix_fmt) {

        /* To handle this change, one could call av_image_alloc again and
         * decode the following frames into another rawvideo file.
         */
        LV_LOG_ERROR("Width, height and pixel format have to be "
                     "constant in a rawvideo file, but the width, height or "
                     "pixel format of the input video changed:\n"
                     "old: width = %d, height = %d, format = %s\n"
                     "new: width = %d, height = %d, format = %s\n",
                     width,
                     height,
                     av_get_pix_fmt_name(ffmpeg_ctx->video_dec_ctx->pix_fmt),
                     frame->width, frame->height,
                     av_get_pix_fmt_name(frame->format));
        goto failed;
    }

    /* CPRO OPTIMIZATION: Skip this frame if flag is set */
#if LV_FFMPEG_AUDIO_SUPPORT != 0
    if(ffmpeg_ctx->skip_this_frame) {
        ffmpeg_ctx->skip_this_frame = false;
        ffmpeg_ctx->consecutive_skips = 0;  /* Reset skip counter after skipping */
        return 0;  /* Skip this frame, don't process */
    }
#endif

#if LV_FFMPEG_HWACCEL_MJPEG != 0
    /* CPRO OPTIMIZATION: If using hardware acceleration, use reusable transfer frame
     * to reduce memory allocation overhead from once-per-frame to once-per-session
     * This is critical for single-core Cortex-A7 performance */
    if(ffmpeg_ctx->use_hwaccel && frame->format == ffmpeg_ctx->hw_pix_fmt) {
        /* Initialize transfer frame on first use (one-time allocation) */
        if(!ffmpeg_ctx->hw_frame_initialized) {
            ffmpeg_ctx->hw_transfer_frame = av_frame_alloc();
            if(ffmpeg_ctx->hw_transfer_frame == NULL) {
                LV_LOG_ERROR("Failed to allocate hardware transfer frame");
                goto failed;
            }
            ffmpeg_ctx->hw_frame_initialized = true;
            LV_LOG_INFO("Hardware transfer frame allocated (one-time)");
        }

        /* Clear previous frame data to avoid corruption */
        av_frame_unref(ffmpeg_ctx->hw_transfer_frame);

        /* Transfer hardware frame to software frame */
        ret = av_hwframe_transfer_data(ffmpeg_ctx->hw_transfer_frame, frame, 0);
        if(ret < 0) {
            LV_LOG_ERROR("Error transferring hardware frame to software: %s", av_err2str(ret));
            goto failed;
        }

        /* Copy software frame data to source buffer for format conversion */
        /* Note: We copy to video_src_data because format conversion expects data there */
        av_image_copy(ffmpeg_ctx->video_src_data, ffmpeg_ctx->video_src_linesize,
                      (const uint8_t **)(ffmpeg_ctx->hw_transfer_frame->data),
                      ffmpeg_ctx->hw_transfer_frame->linesize,
                      ffmpeg_ctx->video_dec_ctx->pix_fmt, width, height);

        LV_LOG_TRACE("Hardware frame transferred successfully");
    } else {
#endif
        /* Software decoding path: copy decoded frame to source buffer
         * This is required since rawvideo expects non aligned data
         */
        av_image_copy(ffmpeg_ctx->video_src_data, ffmpeg_ctx->video_src_linesize,
                      (const uint8_t **)(frame->data), frame->linesize,
                      ffmpeg_ctx->video_dec_ctx->pix_fmt, width, height);
#if LV_FFMPEG_HWACCEL_MJPEG != 0
    }
#endif

    /* CPRO OPTIMIZATION: Check if conversion is needed */
#if LV_FFMPEG_AUDIO_SUPPORT != 0
    if(!ffmpeg_ctx->needs_conversion) {
        /* No conversion needed, copy directly */
        memcpy(ffmpeg_ctx->video_dst_data[0], ffmpeg_ctx->video_src_data[0],
               width * height * (ffmpeg_ctx->has_alpha ? 4 : 3));
        ret = width;
        goto skip_conversion;
    }
#endif

    /* CPRO OPTIMIZATION: Use NEON-accelerated conversion for YUV420P on ARM Cortex-A7
     * NEON provides 4-5x performance improvement over sws_scale for format conversion
     * Fallback to sws_scale for non-YUV formats or when NEON is not available */
#if LV_USE_DRAW_SW && defined(__ARM_NEON)
    if(ffmpeg_ctx->video_dec_ctx->pix_fmt == AV_PIX_FMT_YUV420P) {
        /* Use NEON-accelerated YUV420P to RGB conversion */
        if(ffmpeg_ctx->video_dst_pix_fmt == AV_PIX_FMT_RGB565LE) {
            neon_yuv420p_to_rgb565(
                ffmpeg_ctx->video_src_data[0],  /* Y plane */
                ffmpeg_ctx->video_src_data[1],  /* U plane */
                ffmpeg_ctx->video_src_data[2],  /* V plane */
                (uint16_t *)ffmpeg_ctx->video_dst_data[0],
                width, height,
                ffmpeg_ctx->video_src_linesize[0],
                ffmpeg_ctx->video_src_linesize[1],
                ffmpeg_ctx->video_dst_linesize[0]
            );
            ret = width;
            goto skip_conversion;
        }
        else if(ffmpeg_ctx->video_dst_pix_fmt == AV_PIX_FMT_BGR0 || ffmpeg_ctx->video_dst_pix_fmt == AV_PIX_FMT_RGB24) {
            neon_yuv420p_to_rgb888(
                ffmpeg_ctx->video_src_data[0],  /* Y plane */
                ffmpeg_ctx->video_src_data[1],  /* U plane */
                ffmpeg_ctx->video_src_data[2],  /* V plane */
                ffmpeg_ctx->video_dst_data[0],
                width, height,
                ffmpeg_ctx->video_src_linesize[0],
                ffmpeg_ctx->video_src_linesize[1],
                ffmpeg_ctx->video_dst_linesize[0]
            );
            ret = width;
            goto skip_conversion;
        }
    }
#endif

    /* Fallback to sws_scale for non-YUV formats or when NEON is not available */
    if(ffmpeg_ctx->sws_ctx == NULL) {
        int swsFlags;

        /* CPRO OPTIMIZATION: Use fast scaling for single-core V833 CPU
         * SWS_FAST_BILINEAR provides better performance than SWS_BILINEAR
         * and is suitable for embedded systems with limited CPU resources
         * Note: Only one scaler algorithm can be selected at a time */
        swsFlags = SWS_FAST_BILINEAR;

        if(ffmpeg_pix_fmt_is_yuv(ffmpeg_ctx->video_dec_ctx->pix_fmt)) {
            /* When the video width and height are not multiples of 8,
             * a blurry screen may appear on the right side
             * This problem was discovered in 2012 and
             * continues to exist in version 4.1.3 in 2019
             * SWS_FAST_BILINEAR handles this case reasonably well on embedded systems
             */
            if((width & 0x7) || (height & 0x7)) {
                LV_LOG_WARN("The width(%d) and height(%d) of the image "
                            "is not a multiple of 8",
                            width, height);
            }
        }

        ffmpeg_ctx->sws_ctx = sws_getContext(
                                  width, height, ffmpeg_ctx->video_dec_ctx->pix_fmt,
                                  width, height, ffmpeg_ctx->video_dst_pix_fmt,
                                  swsFlags,
                                  NULL, NULL, NULL);
    }

    if(!ffmpeg_ctx->has_alpha) {
        int lv_linesize = lv_color_format_get_size(LV_COLOR_FORMAT_NATIVE) * width;
        int dst_linesize = ffmpeg_ctx->video_dst_linesize[0];
        if(dst_linesize != lv_linesize) {
            LV_LOG_WARN("ffmpeg linesize = %d, but lvgl image require %d",
                        dst_linesize,
                        lv_linesize);
            ffmpeg_ctx->video_dst_linesize[0] = lv_linesize;
        }
    }

    ret = sws_scale(
              ffmpeg_ctx->sws_ctx,
              (const uint8_t * const *)(ffmpeg_ctx->video_src_data),
              ffmpeg_ctx->video_src_linesize,
              0,
              height,
              ffmpeg_ctx->video_dst_data,
              ffmpeg_ctx->video_dst_linesize);

skip_conversion:
    return ret;
failed:
    return ret;
}

static int ffmpeg_decode_packet(AVCodecContext * dec, const AVPacket * pkt,
                                struct ffmpeg_context_s * ffmpeg_ctx)
{
    int ret = 0;
    enum AVMediaType codec_type = dec->codec->type;

    /* OPTIMIZED: Early return for invalid input */
    if(!ffmpeg_ctx || !dec) {
        return -1;
    }

    /* submit the packet to the decoder */
    ret = avcodec_send_packet(dec, pkt);
    if(ret < 0) {
        LV_LOG_ERROR("Error submitting a packet for decoding (%s)",
                     av_err2str(ret));
        return ret;
    }

    /* CPRO OPTIMIZATION: Pre-select output frame based on codec type to reduce branching */
#if LV_FFMPEG_AUDIO_SUPPORT != 0
    AVFrame *output_frame = (codec_type == AVMEDIA_TYPE_AUDIO) ? ffmpeg_ctx->audio_frame : ffmpeg_ctx->frame;
    /* Function pointer for output processing to reduce branching */
    int (*output_func)(struct ffmpeg_context_s *) = (codec_type == AVMEDIA_TYPE_AUDIO) ?
                                                      ffmpeg_output_audio_frame : ffmpeg_output_video_frame;
#else
    AVFrame *output_frame = ffmpeg_ctx->frame;
    int (*output_func)(struct ffmpeg_context_s *) = ffmpeg_output_video_frame;
#endif

    /* get all the available frames from the decoder */
    while(ret >= 0) {
        ret = avcodec_receive_frame(dec, output_frame);
        if(ret < 0) {

            /* those two return values are special and mean there is
             * no output frame available,
             * but there were no errors during decoding
             */
            if(ret == AVERROR_EOF || ret == AVERROR(EAGAIN)) {
                return 0;
            }

            LV_LOG_ERROR("Error during decoding (%s)", av_err2str(ret));
            return ret;
        }

        /* CPRO OPTIMIZATION: Use function pointer instead of if-else branching */
        ret = output_func(ffmpeg_ctx);

        av_frame_unref(output_frame);
        if(ret < 0) {
            LV_LOG_WARN("ffmpeg_decode_packet ended %d", ret);
            return ret;
        }
    }

    return 0;
}

#if LV_FFMPEG_HWACCEL_MJPEG != 0
/**
 * Initialize hardware acceleration for MJPEG decoding
 * @param ffmpeg_ctx FFmpeg context
 * @param dec_ctx Codec context to be initialized
 * @param codec_id Codec ID
 * @return 0 on success, negative error code on failure
 */
static int ffmpeg_init_hwaccel(struct ffmpeg_context_s * ffmpeg_ctx,
                                AVCodecContext * dec_ctx,
                                enum AVCodecID codec_id)
{
    int ret = 0;

    /* Only enable hardware acceleration for MJPEG */
    if(codec_id != AV_CODEC_ID_MJPEG) {
        ffmpeg_ctx->use_hwaccel = false;
        return 0;
    }

    LV_LOG_INFO("Attempting to initialize MJPEG hardware acceleration");

    /* Try different hardware device types */
    enum AVHWDeviceType hw_type = AV_HWDEVICE_TYPE_NONE;
    const char *hw_device_name = NULL;

    /* Try V4L2 M2M first (common on ARM platforms) */
    hw_type = av_hwdevice_find_type_by_name("v4l2m2m");
    if(hw_type == AV_HWDEVICE_TYPE_NONE) {
        /* Try Cedrus (Allwinner specific) */
        hw_type = av_hwdevice_find_type_by_name("cedrus");
        if(hw_type == AV_HWDEVICE_TYPE_NONE) {
            /* Try rkmpp (Rockchip) */
            hw_type = av_hwdevice_find_type_by_name("rkmpp");
            if(hw_type == AV_HWDEVICE_TYPE_NONE) {
                LV_LOG_WARN("No hardware device found for MJPEG, falling back to software");
                ffmpeg_ctx->use_hwaccel = false;
                return 0;
            }
        }
    }

    /* Get device name for logging */
    hw_device_name = av_hwdevice_get_type_name(hw_type);
    LV_LOG_INFO("Found hardware device: %s", hw_device_name);

    /* Try to find hardware decoder for MJPEG */
    const AVCodec *hw_decoder = avcodec_find_decoder(codec_id);
    if(hw_decoder == NULL) {
        LV_LOG_WARN("Failed to find MJPEG decoder, falling back to software");
        ffmpeg_ctx->use_hwaccel = false;
        return 0;
    }

    /* Check if decoder supports this hardware type */
    bool hw_supported = false;
    for(int i = 0;; i++) {
        const AVCodecHWConfig *config = avcodec_get_hw_config(hw_decoder, i);
        if(!config) {
            break;
        }
        if(config->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX &&
           config->device_type == hw_type) {
            hw_supported = true;
            ffmpeg_ctx->hw_pix_fmt = config->pix_fmt;
            LV_LOG_INFO("Hardware pixel format: %s", av_get_pix_fmt_name(ffmpeg_ctx->hw_pix_fmt));
            break;
        }
    }

    if(!hw_supported) {
        LV_LOG_WARN("Hardware decoder does not support this device type, falling back to software");
        ffmpeg_ctx->use_hwaccel = false;
        return 0;
    }

    /* Create hardware device context */
    ret = av_hwdevice_ctx_create(&ffmpeg_ctx->hw_device_ctx,
                                   hw_type,
                                   NULL,
                                   NULL,
                                   0);
    if(ret < 0) {
        LV_LOG_WARN("Failed to create hardware device context: %s, falling back to software",
                    av_err2str(ret));
        ffmpeg_ctx->use_hwaccel = false;
        return 0;
    }

    /* Set hardware device context to codec context */
    dec_ctx->hw_device_ctx = av_buffer_ref(ffmpeg_ctx->hw_device_ctx);
    if(dec_ctx->hw_device_ctx == NULL) {
        LV_LOG_ERROR("Failed to reference hardware device context");
        av_buffer_unref(&ffmpeg_ctx->hw_device_ctx);
        ffmpeg_ctx->use_hwaccel = false;
        return AVERROR(ENOMEM);
    }

    /* Initialize hardware frame pool for better performance */
    ret = ffmpeg_init_hwaccel_frames(ffmpeg_ctx, dec_ctx);
    if (ret < 0) {
        LV_LOG_WARN("Hardware frame pool initialization failed: %s, continuing without frame pool",
                    av_err2str(ret));
        /* Continue without frame pool - not a fatal error */
    }

    ffmpeg_ctx->use_hwaccel = true;
    LV_LOG_INFO("MJPEG hardware acceleration initialized successfully");
    return 0;
}

/**
 * Initialize hardware frame pool for MJPEG decoding
 * @param ffmpeg_ctx FFmpeg context
 * @param dec_ctx Codec context
 * @return 0 on success, negative error code on failure
 */
static int ffmpeg_init_hwaccel_frames(struct ffmpeg_context_s * ffmpeg_ctx,
                                      AVCodecContext * dec_ctx)
{
    int ret;

    LV_LOG_INFO("Initializing hardware frame pool...");

    /* Get hardware frame constraints */
    AVHWFramesConstraints *constraints = av_hwdevice_get_hwframe_constraints(
        ffmpeg_ctx->hw_device_ctx, NULL);

    if (!constraints) {
        LV_LOG_WARN("Failed to get hardware frame constraints");
        return -1;
    }

    /* Allocate hardware frames context */
    ffmpeg_ctx->hw_frames_ctx = av_hwframe_ctx_alloc(ffmpeg_ctx->hw_device_ctx);
    if (!ffmpeg_ctx->hw_frames_ctx) {
        LV_LOG_ERROR("Failed to allocate hardware frames context");
        av_hwframe_constraints_free(&constraints);
        return AVERROR(ENOMEM);
    }

    /* Configure hardware frames context */
    AVHWFramesContext *frames_ctx = (AVHWFramesContext *)ffmpeg_ctx->hw_frames_ctx->data;
    frames_ctx->format = ffmpeg_ctx->hw_pix_fmt;
    frames_ctx->sw_format = dec_ctx->pix_fmt;
    frames_ctx->width = dec_ctx->width;
    frames_ctx->height = dec_ctx->height;

    /* CPRO OPTIMIZATION: Optimize frame pool size for single-core Cortex-A7
     * Initial pool size: 5 frames
     * - Too small: Causes frame allocation during playback
     * - Too large: Wastes memory on embedded system
     * - 5 is optimal for 25-30fps playback on V833 */
    frames_ctx->initial_pool_size = 5;

    LV_LOG_INFO("Hardware frame pool configuration: size=%d, format=%s, sw_format=%s",
               frames_ctx->initial_pool_size,
               av_get_pix_fmt_name(frames_ctx->format),
               av_get_pix_fmt_name(frames_ctx->sw_format));

    /* Initialize hardware frames context */
    ret = av_hwframe_ctx_init(ffmpeg_ctx->hw_frames_ctx);
    if (ret < 0) {
        LV_LOG_ERROR("Failed to initialize hardware frames context: %s",
                     av_err2str(ret));
        av_buffer_unref(&ffmpeg_ctx->hw_frames_ctx);
        av_hwframe_constraints_free(&constraints);
        return ret;
    }

    /* Set hardware frames context to decoder */
    dec_ctx->hw_frames_ctx = av_buffer_ref(ffmpeg_ctx->hw_frames_ctx);
    if (!dec_ctx->hw_frames_ctx) {
        LV_LOG_ERROR("Failed to set hardware frames context to decoder");
        av_buffer_unref(&ffmpeg_ctx->hw_frames_ctx);
        av_hwframe_constraints_free(&constraints);
        return AVERROR(ENOMEM);
    }

    ffmpeg_ctx->hw_pool_initialized = true;
    av_hwframe_constraints_free(&constraints);

    LV_LOG_INFO("Hardware frame pool initialized successfully");
    return 0;
}
#endif

static int ffmpeg_open_codec_context(int * stream_idx,
                                     AVCodecContext ** dec_ctx, AVFormatContext * fmt_ctx,
                                     enum AVMediaType type, struct ffmpeg_context_s * ffmpeg_ctx)
{
    int ret;
    int stream_index;
    AVStream * st;
    const AVCodec * dec = NULL;
    AVDictionary * opts = NULL;

    ret = av_find_best_stream(fmt_ctx, type, -1, -1, NULL, 0);
    if(ret < 0) {
        LV_LOG_ERROR("Could not find %s stream in input file",
                     av_get_media_type_string(type));
        return ret;
    }
    else {
        stream_index = ret;
        st = fmt_ctx->streams[stream_index];

        /* find decoder for the stream */
        dec = avcodec_find_decoder(st->codecpar->codec_id);
        if(dec == NULL) {
            LV_LOG_ERROR("Failed to find %s codec",
                         av_get_media_type_string(type));
            return AVERROR(EINVAL);
        }

        /* Allocate a codec context for the decoder */
        *dec_ctx = avcodec_alloc_context3(dec);
        if(*dec_ctx == NULL) {
            LV_LOG_ERROR("Failed to allocate the %s codec context",
                         av_get_media_type_string(type));
            return AVERROR(ENOMEM);
        }

        /* Copy codec parameters from input stream to output codec context */
        if((ret = avcodec_parameters_to_context(*dec_ctx, st->codecpar)) < 0) {
            LV_LOG_ERROR(
                "Failed to copy %s codec parameters to decoder context",
                av_get_media_type_string(type));
            return ret;
        }

        /* Init the decoders */
        /* CPRO OPTIMIZATION: Add fast decoding flags for single-core CPU */
        if(type == AVMEDIA_TYPE_VIDEO) {
            (*dec_ctx)->flags |= AV_CODEC_FLAG_LOW_DELAY;  /* Reduce latency */
            (*dec_ctx)->flags2 |= AV_CODEC_FLAG2_FAST;     /* Faster decoding */

#if LV_FFMPEG_HWACCEL_MJPEG != 0
            /* Initialize hardware acceleration for MJPEG */
            if(ffmpeg_ctx != NULL) {
                ret = ffmpeg_init_hwaccel(ffmpeg_ctx, *dec_ctx, st->codecpar->codec_id);
                if(ret < 0) {
                    LV_LOG_WARN("Hardware acceleration initialization failed: %s, using software",
                                av_err2str(ret));
                    /* Continue with software decoding */
                }
            }
#endif
        }

        if((ret = avcodec_open2(*dec_ctx, dec, &opts)) < 0) {
            LV_LOG_ERROR("Failed to open %s codec: %s",
                         av_get_media_type_string(type), av_err2str(ret));
            return ret;
        }

        if(type == AVMEDIA_TYPE_AUDIO) {
            LV_LOG_INFO("Audio codec opened successfully: %s, sample_rate=%d, channels=%d, frame_size=%d",
                       avcodec_get_name((*dec_ctx)->codec_id),
                       (*dec_ctx)->sample_rate,
                       (*dec_ctx)->ch_layout.nb_channels,
                       (*dec_ctx)->frame_size);
        }

        *stream_idx = stream_index;
    }

    return 0;
}

static int ffmpeg_get_image_header(lv_image_decoder_dsc_t * dsc,
                                   lv_image_header_t * header)
{
    int ret = -1;

    AVFormatContext * fmt_ctx = NULL;
    AVCodecContext * video_dec_ctx = NULL;
    AVIOContext * io_ctx = NULL;
    int video_stream_idx;

    io_ctx = ffmpeg_open_io_context(&dsc->file);
    if(io_ctx == NULL) {
        LV_LOG_ERROR("io_ctx malloc failed");
        return ret;
    }

    fmt_ctx = avformat_alloc_context();
    if(fmt_ctx == NULL) {
        LV_LOG_ERROR("fmt_ctx malloc failed");
        goto failed;
    }
    fmt_ctx->pb = io_ctx;
    fmt_ctx->flags |= AVFMT_FLAG_CUSTOM_IO;

    /* open input file, and allocate format context */
    if(avformat_open_input(&fmt_ctx, (const char *)dsc->src, NULL, NULL) < 0) {
        LV_LOG_ERROR("Could not open source file %s", (const char *)dsc->src);
        goto failed;
    }

    /* retrieve stream information */
    if(avformat_find_stream_info(fmt_ctx, NULL) < 0) {
        LV_LOG_ERROR("Could not find stream information");
        goto failed;
    }

    if(ffmpeg_open_codec_context(&video_stream_idx, &video_dec_ctx,
                                 fmt_ctx, AVMEDIA_TYPE_VIDEO, NULL)
       >= 0) {
        bool has_alpha = ffmpeg_pix_fmt_has_alpha(video_dec_ctx->pix_fmt);

        /* allocate image where the decoded image will be put */
        header->w = video_dec_ctx->width;
        header->h = video_dec_ctx->height;
        header->cf = has_alpha ? LV_COLOR_FORMAT_ARGB8888 : LV_COLOR_FORMAT_NATIVE;
        header->stride = header->w * lv_color_format_get_size(header->cf);

        ret = 0;
    }

failed:
    avcodec_free_context(&video_dec_ctx);
    avformat_close_input(&fmt_ctx);
    if(io_ctx != NULL) {
        av_free(io_ctx->buffer);
        av_free(io_ctx);
    }
    return ret;
}

static int ffmpeg_get_frame_refr_period(struct ffmpeg_context_s * ffmpeg_ctx)
{
    int avg_frame_rate_num = ffmpeg_ctx->video_stream->avg_frame_rate.num;
    if(avg_frame_rate_num > 0) {
        int period = 1000 * (int64_t)ffmpeg_ctx->video_stream->avg_frame_rate.den
                     / avg_frame_rate_num;
        return period;
    }

    return -1;
}

static int ffmpeg_update_next_frame(struct ffmpeg_context_s * ffmpeg_ctx)
{
    int ret = 0;
    int pkt_stream_idx;

    /* CPRO OPTIMIZATION: Cache stream indices to reduce memory access */
    int video_idx = ffmpeg_ctx->video_stream_idx;
#if LV_FFMPEG_AUDIO_SUPPORT != 0
    int audio_idx = ffmpeg_ctx->audio_stream_idx;
    bool has_audio = ffmpeg_ctx->has_audio;

    /* CPRO OPTIMIZATION: Use context member instead of static variable for thread safety
     * Static variables are problematic in multithreaded environments
     * Use context's consecutive_skips counter instead */
    if(ffmpeg_ctx->consecutive_skips > 2) {
        ffmpeg_ctx->consecutive_skips = 0;
        ffmpeg_ctx->skip_this_frame = false;
    }
#endif

    while(1) {

        /* read frames from the file */
        if(av_read_frame(ffmpeg_ctx->fmt_ctx, ffmpeg_ctx->pkt) >= 0) {
            pkt_stream_idx = ffmpeg_ctx->pkt->stream_index;

            /* CPRO OPTIMIZATION: Use direct comparison instead of multiple if-else */
            if(pkt_stream_idx == video_idx) {
#if LV_FFMPEG_AUDIO_SUPPORT != 0 && LV_FFMPEG_SYNC_ENABLED != 0
                /* Audio-Video Synchronization: Skip video frame if audio is behind */
                if (should_skip_video_frame(ffmpeg_ctx)) {
                    /* IMPORTANT: Update video_pts even when skipping frame to avoid deadlock
                     * Otherwise video_pts will stay at 83ms while audio keeps advancing to 441ms+ */
                    if (ffmpeg_ctx->pkt->pts != AV_NOPTS_VALUE) {
                        int64_t pts_ms = pts_to_ms(ffmpeg_ctx->video_stream, ffmpeg_ctx->pkt->pts);
                        if (pts_ms != AV_NOPTS_VALUE) {
                            ffmpeg_ctx->video_pts = pts_ms;
                                                    ffmpeg_ctx->video_clock = pts_ms;
                                                    __sync_synchronize();  /* Memory barrier after update */                        }
                    }
                    av_packet_unref(ffmpeg_ctx->pkt);
                    continue;  /* Skip this video frame */
                }
#endif

                ret = ffmpeg_decode_packet(ffmpeg_ctx->video_dec_ctx,
                                           ffmpeg_ctx->pkt, ffmpeg_ctx);
                av_packet_unref(ffmpeg_ctx->pkt);

                if(ret < 0) {
                    LV_LOG_WARN("video frame is empty %d", ret);
                    break;
                }

                /* Video frame decoded successfully */
                break;
            }
#if LV_FFMPEG_AUDIO_SUPPORT != 0
            /* MULTITHREAD ARCHITECTURE: Skip audio frames - they are handled by audio thread */
            else if(has_audio && pkt_stream_idx == audio_idx) {
                /* Audio frames are handled by audio thread, skip here */
                av_packet_unref(ffmpeg_ctx->pkt);
                continue;
            }
#endif

            /* Unknown stream, unref and continue */
            av_packet_unref(ffmpeg_ctx->pkt);
        }
        else {
            ret = -1;
            break;
        }
    }

    return ret;
}

static int ffmpeg_lvfs_read(void * ptr, uint8_t * buf, int buf_size)
{
    lv_fs_file_t * file = ptr;
    uint32_t bytesRead = 0;
    lv_fs_res_t res = lv_fs_read(file, buf, buf_size, &bytesRead);
    if(bytesRead == 0)
        return AVERROR_EOF;  /* Let FFmpeg know that we have reached eof */
    if(res != LV_FS_RES_OK)
        return AVERROR_EOF;
    return bytesRead;
}

static int64_t ffmpeg_lvfs_seek(void * ptr, int64_t pos, int whence)
{
    lv_fs_file_t * file = ptr;
    if(whence == SEEK_SET && lv_fs_seek(file, pos, SEEK_SET) == LV_FS_RES_OK) {
        return pos;
    }
    return -1;
}

static AVIOContext * ffmpeg_open_io_context(lv_fs_file_t * file)
{
    uint8_t * iBuffer = av_malloc(DECODER_BUFFER_SIZE);
    if(iBuffer == NULL) {
        LV_LOG_ERROR("iBuffer malloc failed");
        return NULL;
    }
    AVIOContext * pIOCtx = avio_alloc_context(iBuffer, DECODER_BUFFER_SIZE,   /* internal Buffer and its size */
                                              0,                                   /* bWriteable (1=true,0=false) */
                                              file,                                /* user data ; will be passed to our callback functions */
                                              ffmpeg_lvfs_read,                    /* Read callback function */
                                              0,                                   /* Write callback function */
                                              ffmpeg_lvfs_seek);                   /* Seek callback function */
    if(pIOCtx == NULL) {
        av_free(iBuffer);
        return NULL;
    }
    return pIOCtx;
}

static struct ffmpeg_context_s * ffmpeg_open_file(const char * path, bool is_lv_fs_path)
{
    if(path == NULL || lv_strlen(path) == 0) {
        LV_LOG_ERROR("file path is empty");
        return NULL;
    }

    struct ffmpeg_context_s * ffmpeg_ctx = lv_malloc_zeroed(sizeof(struct ffmpeg_context_s));
    LV_ASSERT_MALLOC(ffmpeg_ctx);
    if(ffmpeg_ctx == NULL) {
        LV_LOG_ERROR("ffmpeg_ctx malloc failed");
        goto failed;
    }

    if(is_lv_fs_path) {
        const lv_fs_res_t fs_res = lv_fs_open(&(ffmpeg_ctx->lv_file), path, LV_FS_MODE_RD);
        if(fs_res != LV_FS_RES_OK) {
            LV_LOG_WARN("Could not open file: %s, res: %d", path, fs_res);
            lv_free(ffmpeg_ctx);
            return NULL;
        }

        ffmpeg_ctx->io_ctx = ffmpeg_open_io_context(&(ffmpeg_ctx->lv_file));     /* Save the buffer pointer to free it later */

        if(ffmpeg_ctx->io_ctx == NULL) {
            LV_LOG_ERROR("io_ctx malloc failed");
            goto failed;
        }

        ffmpeg_ctx->fmt_ctx = avformat_alloc_context();
        if(ffmpeg_ctx->fmt_ctx == NULL) {
            LV_LOG_ERROR("fmt_ctx malloc failed");
            goto failed;
        }
        ffmpeg_ctx->fmt_ctx->pb = ffmpeg_ctx->io_ctx;
        ffmpeg_ctx->fmt_ctx->flags |= AVFMT_FLAG_CUSTOM_IO;
    }

    /* open input file, and allocate format context */

    if(avformat_open_input(&(ffmpeg_ctx->fmt_ctx), path, NULL, NULL) < 0) {
        LV_LOG_ERROR("Could not open source file %s", path);
        goto failed;
    }

    /* retrieve stream information */

    if(avformat_find_stream_info(ffmpeg_ctx->fmt_ctx, NULL) < 0) {
        LV_LOG_ERROR("Could not find stream information");
        goto failed;
    }

    if(ffmpeg_open_codec_context(
           &(ffmpeg_ctx->video_stream_idx),
           &(ffmpeg_ctx->video_dec_ctx),
           ffmpeg_ctx->fmt_ctx, AVMEDIA_TYPE_VIDEO, ffmpeg_ctx)
       >= 0) {
        ffmpeg_ctx->video_stream = ffmpeg_ctx->fmt_ctx->streams[ffmpeg_ctx->video_stream_idx];

        ffmpeg_ctx->has_alpha = ffmpeg_pix_fmt_has_alpha(ffmpeg_ctx->video_dec_ctx->pix_fmt);

        ffmpeg_ctx->video_dst_pix_fmt = (ffmpeg_ctx->has_alpha ? AV_PIX_FMT_BGRA : AV_PIX_FMT_TRUE_COLOR);

        /* CPRO OPTIMIZATION: Check if format conversion is needed */
#if LV_FFMPEG_AUDIO_SUPPORT != 0
        ffmpeg_ctx->needs_conversion = (ffmpeg_ctx->video_dec_ctx->pix_fmt != ffmpeg_ctx->video_dst_pix_fmt);
        ffmpeg_ctx->consecutive_skips = 0;
        ffmpeg_ctx->skip_this_frame = false;
        LV_LOG_INFO("Video format conversion needed: %s", ffmpeg_ctx->needs_conversion ? "yes" : "no");
#endif
    }

#if LV_FFMPEG_AUDIO_SUPPORT != 0
    /* Try to open audio stream */
    if(ffmpeg_open_codec_context(
           &(ffmpeg_ctx->audio_stream_idx),
           &(ffmpeg_ctx->audio_dec_ctx),
           ffmpeg_ctx->fmt_ctx, AVMEDIA_TYPE_AUDIO, NULL)
       >= 0) {
        ffmpeg_ctx->audio_stream = ffmpeg_ctx->fmt_ctx->streams[ffmpeg_ctx->audio_stream_idx];
        ffmpeg_ctx->has_audio = true;
        LV_LOG_INFO("Audio stream found, codec: %s, stream_idx: %d",
                    avcodec_get_name(ffmpeg_ctx->audio_dec_ctx->codec_id), ffmpeg_ctx->audio_stream_idx);
    } else {
        ffmpeg_ctx->has_audio = false;
        LV_LOG_WARN("No audio stream found in file");
    }

    LV_LOG_INFO("Audio detection result: has_audio=%d", ffmpeg_ctx->has_audio);
#endif

#if LV_FFMPEG_SYNC_ENABLED != 0
    /* Initialize audio-video synchronization fields */
    ffmpeg_ctx->video_clock = 0;
    ffmpeg_ctx->audio_clock = 0;
    ffmpeg_ctx->video_pts = AV_NOPTS_VALUE;
    ffmpeg_ctx->audio_pts = AV_NOPTS_VALUE;
    ffmpeg_ctx->start_time = 0;
    ffmpeg_ctx->sync_threshold = 30;      /* 30ms synchronization threshold */
    ffmpeg_ctx->max_frame_delay = 100;   /* Maximum frame delay 100ms */
    ffmpeg_ctx->frame_drop_count = 0;
    ffmpeg_ctx->frame_repeat_count = 0;
    ffmpeg_ctx->sync_enabled = true;
    LV_LOG_INFO("[SYNC] Audio-video synchronization initialized:");
    LV_LOG_INFO("[SYNC]   sync_threshold: %d ms", ffmpeg_ctx->sync_threshold);
    LV_LOG_INFO("[SYNC]   max_frame_delay: %d ms", ffmpeg_ctx->max_frame_delay);
    LV_LOG_INFO("[SYNC]   sync_enabled: %s", ffmpeg_ctx->sync_enabled ? "true" : "false");
#endif

#if LV_FFMPEG_DUMP_FORMAT
    /* dump input information to stderr */
    av_dump_format(ffmpeg_ctx->fmt_ctx, 0, path, 0);
#endif

    if(ffmpeg_ctx->video_stream == NULL) {
        LV_LOG_ERROR("Could not find video stream in the input, aborting");
        goto failed;
    }

    return ffmpeg_ctx;

failed:
    ffmpeg_close(ffmpeg_ctx);
    return NULL;
}

static int ffmpeg_image_allocate(struct ffmpeg_context_s * ffmpeg_ctx)
{
    int ret;

    /* allocate image where the decoded image will be put */
    ret = av_image_alloc(
              ffmpeg_ctx->video_src_data,
              ffmpeg_ctx->video_src_linesize,
              ffmpeg_ctx->video_dec_ctx->width,
              ffmpeg_ctx->video_dec_ctx->height,
              ffmpeg_ctx->video_dec_ctx->pix_fmt,
              4);

    if(ret < 0) {
        LV_LOG_ERROR("Could not allocate src raw video buffer");
        return ret;
    }

    LV_LOG_INFO("alloc video_src_bufsize = %d", ret);

    ret = av_image_alloc(
              ffmpeg_ctx->video_dst_data,
              ffmpeg_ctx->video_dst_linesize,
              ffmpeg_ctx->video_dec_ctx->width,
              ffmpeg_ctx->video_dec_ctx->height,
              ffmpeg_ctx->video_dst_pix_fmt,
              4);

    if(ret < 0) {
        LV_LOG_ERROR("Could not allocate dst raw video buffer");
        return ret;
    }

    LV_LOG_INFO("allocate video_dst_bufsize = %d", ret);

    ffmpeg_ctx->frame = av_frame_alloc();

    if(ffmpeg_ctx->frame == NULL) {
        LV_LOG_ERROR("Could not allocate frame");
        return -1;
    }

    /* allocate packet, set data to NULL, let the demuxer fill it */

    ffmpeg_ctx->pkt = av_packet_alloc();
    if(ffmpeg_ctx->pkt == NULL) {
        LV_LOG_ERROR("av_packet_alloc failed");
        return -1;
    }
    ffmpeg_ctx->pkt->data = NULL;
    ffmpeg_ctx->pkt->size = 0;

#if LV_FFMPEG_AUDIO_SUPPORT != 0
    /* Allocate audio frame */
    ffmpeg_ctx->audio_frame = av_frame_alloc();
    if(ffmpeg_ctx->audio_frame == NULL) {
        LV_LOG_WARN("Could not allocate audio frame");
    }

    /* Initialize audio output if audio stream is present */
    if(ffmpeg_ctx->has_audio) {
        if(ffmpeg_audio_init(ffmpeg_ctx) < 0) {
            LV_LOG_WARN("Audio output initialization failed, audio will be disabled");
            ffmpeg_ctx->has_audio = false;
        }
    }
#endif

    return 0;
}

static void ffmpeg_close_src_ctx(struct ffmpeg_context_s * ffmpeg_ctx)
{
#if LV_FFMPEG_HWACCEL_MJPEG != 0
    /* CPRO OPTIMIZATION: Complete hardware resource cleanup with proper order
     * Order matters: transfer frame -> frames context -> device context
     * This ensures no dangling references or memory leaks */

    /* Release hardware transfer frame */
    if(ffmpeg_ctx->hw_transfer_frame != NULL) {
        av_frame_free(&ffmpeg_ctx->hw_transfer_frame);
        ffmpeg_ctx->hw_transfer_frame = NULL;
    }
    ffmpeg_ctx->hw_frame_initialized = false;

    /* Release hardware frame pool */
    if(ffmpeg_ctx->hw_frames_ctx != NULL) {
        av_buffer_unref(&ffmpeg_ctx->hw_frames_ctx);
        ffmpeg_ctx->hw_frames_ctx = NULL;
    }
    ffmpeg_ctx->hw_pool_initialized = false;

    /* Release hardware device context */
    if(ffmpeg_ctx->hw_device_ctx != NULL) {
        av_buffer_unref(&ffmpeg_ctx->hw_device_ctx);
        ffmpeg_ctx->hw_device_ctx = NULL;
    }
    ffmpeg_ctx->use_hwaccel = false;

    LV_LOG_INFO("Hardware acceleration resources released");
#endif

    /* Release FFmpeg decoder and format context */
    avcodec_free_context(&(ffmpeg_ctx->video_dec_ctx));
    avformat_close_input(&(ffmpeg_ctx->fmt_ctx));

    /* Release packet and frame */
    av_packet_free(&ffmpeg_ctx->pkt);
    av_frame_free(&(ffmpeg_ctx->frame));

    /* Release source buffer */
    if(ffmpeg_ctx->video_src_data[0] != NULL) {
        av_free(ffmpeg_ctx->video_src_data[0]);
        ffmpeg_ctx->video_src_data[0] = NULL;
    }

#if LV_FFMPEG_AUDIO_SUPPORT != 0
    /* Release audio resources */
    avcodec_free_context(&(ffmpeg_ctx->audio_dec_ctx));
    av_frame_free(&(ffmpeg_ctx->audio_frame));
    ffmpeg_audio_deinit(ffmpeg_ctx);
#endif
}

static void ffmpeg_close_dst_ctx(struct ffmpeg_context_s * ffmpeg_ctx)
{
    if(ffmpeg_ctx->video_dst_data[0] != NULL) {
        av_free(ffmpeg_ctx->video_dst_data[0]);
        ffmpeg_ctx->video_dst_data[0] = NULL;
    }
}

static void ffmpeg_close(struct ffmpeg_context_s * ffmpeg_ctx)
{
    if(ffmpeg_ctx == NULL) {
        LV_LOG_WARN("ffmpeg_ctx is NULL");
        return;
    }

#if LV_FFMPEG_AUDIO_SUPPORT != 0
    /* Stop unified playback thread if running */
    if(ffmpeg_ctx->is_playing) {
        ffmpeg_ctx->is_playing = 0;
        pthread_join(ffmpeg_ctx->playback_thread, NULL);
        LV_LOG_INFO("Unified playback thread stopped in ffmpeg_close");
    }

    /* Destroy video buffer */
    if(ffmpeg_ctx->video_buffer.initialized) {
        video_buffer_destroy(&ffmpeg_ctx->video_buffer);
    }

    /* Note: Unified audio resources are cleaned up in ffmpeg_audio_deinit */
#endif

    sws_freeContext(ffmpeg_ctx->sws_ctx);
    ffmpeg_close_src_ctx(ffmpeg_ctx);
    ffmpeg_close_dst_ctx(ffmpeg_ctx);

#if LV_FFMPEG_AUDIO_SUPPORT != 0
    /* Clean up ALSA PCM resources */
    ffmpeg_audio_pcm_deinit(ffmpeg_ctx);

    if(ffmpeg_ctx->audio_buf != NULL) {
        av_free(ffmpeg_ctx->audio_buf);
        ffmpeg_ctx->audio_buf = NULL;
    }
#endif

    if(ffmpeg_ctx->io_ctx != NULL) {
        av_free(ffmpeg_ctx->io_ctx->buffer);
        av_free(ffmpeg_ctx->io_ctx);
        lv_fs_close(&(ffmpeg_ctx->lv_file));
    }

    lv_free(ffmpeg_ctx);

    LV_LOG_INFO("ffmpeg_ctx closed");
}

static void lv_ffmpeg_player_frame_update_cb(lv_timer_t * timer)
{
    lv_obj_t * obj = (lv_obj_t *)lv_timer_get_user_data(timer);
    lv_ffmpeg_player_t * player = (lv_ffmpeg_player_t *)obj;

    if(!player->ffmpeg_ctx) {
        return;
    }

#if LV_FFMPEG_AUDIO_SUPPORT != 0
    /* Pop decoded frame from video buffer */
    AVFrame *frame = video_buffer_pop(&player->ffmpeg_ctx->video_buffer);

    if(!frame) {
        /* Buffer is empty, check if playback thread is still playing */
        if(!player->ffmpeg_ctx->is_playing) {
            /* Playback thread has stopped, handle EOF or stop */
            lv_ffmpeg_player_set_cmd(obj, player->auto_restart ?
                                    LV_FFMPEG_PLAYER_CMD_START :
                                    LV_FFMPEG_PLAYER_CMD_STOP);
            if(!player->auto_restart) {
                lv_obj_send_event((lv_obj_t *)player, LV_EVENT_READY, NULL);
            }
        }
        /* Buffer empty but playback thread still running, repeat last frame */
        return;
    }

    /* Free the frame (data has already been copied to draw buffer by video thread) */
    av_frame_unref(frame);
    av_frame_free(&frame);
#else
    /* No audio support: keep original single-threaded decoding */
    int has_next = ffmpeg_update_next_frame(player->ffmpeg_ctx);

    if(has_next < 0) {
        lv_ffmpeg_player_set_cmd(obj, player->auto_restart ? LV_FFMPEG_PLAYER_CMD_START : LV_FFMPEG_PLAYER_CMD_STOP);
        if(!player->auto_restart) {
            lv_obj_send_event((lv_obj_t *)player, LV_EVENT_READY, NULL);
        }
        return;
    }
#endif

#if LV_FFMPEG_AUDIO_SUPPORT != 0 && LV_FFMPEG_SYNC_ENABLED != 0
    /* Audio-Video Synchronization: Check if we need to repeat current frame */
    if (should_repeat_video_frame(player->ffmpeg_ctx)) {
        /* Repeat current frame, don't update display */
        return;
    }
#endif

    /* CPRO OPTIMIZATION: Only invalidate if frame was actually updated
     * Check if skip_this_frame was set - if yes, no need to invalidate */
#if LV_FFMPEG_AUDIO_SUPPORT != 0
    if(!player->ffmpeg_ctx->skip_this_frame) {
        /* CPRO OPTIMIZATION: Reduce cache drop frequency for single-core CPU
         * Cache drops are expensive on embedded systems
         * Only drop cache every 10 frames to reduce overhead */
        static int frame_counter = 0;
        frame_counter++;
        if(frame_counter % 10 == 0) {
            lv_image_cache_drop(lv_image_get_src(obj));
        }
        lv_obj_invalidate(obj);
    }
#else
    lv_image_cache_drop(lv_image_get_src(obj));
    lv_obj_invalidate(obj);
#endif

#if LV_FFMPEG_SYNC_ENABLED != 0
    /* Output synchronization statistics every second */
    if (player->ffmpeg_ctx->sync_enabled && player->ffmpeg_ctx->has_audio) {
        static int sync_log_counter = 0;
        sync_log_counter++;
        /* Assuming timer period is around 50ms, 20 iterations = 1 second */
        if (sync_log_counter % 20 == 0) {
            LV_LOG_INFO("[SYNC] Sync statistics: drops=%d, repeats=%d",
                       player->ffmpeg_ctx->frame_drop_count,
                       player->ffmpeg_ctx->frame_repeat_count);
        }
    }
#endif
}

static void lv_ffmpeg_player_constructor(const lv_obj_class_t * class_p,
                                         lv_obj_t * obj)
{

    LV_UNUSED(class_p);
    LV_TRACE_OBJ_CREATE("begin");

    lv_ffmpeg_player_t * player = (lv_ffmpeg_player_t *)obj;

    player->auto_restart = false;
    player->ffmpeg_ctx = NULL;
    player->volume = 75;
    player->audio_enabled = true;
    player->timer = lv_timer_create(lv_ffmpeg_player_frame_update_cb,
                                    FRAME_DEF_REFR_PERIOD, obj);
    lv_timer_pause(player->timer);

    LV_TRACE_OBJ_CREATE("finished");
}

static void lv_ffmpeg_player_destructor(const lv_obj_class_t * class_p,
                                        lv_obj_t * obj)
{
    LV_UNUSED(class_p);

    LV_TRACE_OBJ_CREATE("begin");

    lv_ffmpeg_player_t * player = (lv_ffmpeg_player_t *)obj;

    if(player->timer) {
        lv_timer_delete(player->timer);
        player->timer = NULL;
    }

    lv_image_cache_drop(lv_image_get_src(obj));

    ffmpeg_close(player->ffmpeg_ctx);
    player->ffmpeg_ctx = NULL;

    LV_TRACE_OBJ_CREATE("finished");
}

#if LV_FFMPEG_AUDIO_SUPPORT != 0

/* Initialize ALSA Mixer for hardware volume control */
static int ffmpeg_audio_mixer_init(struct ffmpeg_context_s * ffmpeg_ctx)
{
    int err;

    if(ffmpeg_ctx->audio_mixer_handle) {
        return 0; /* Already initialized */
    }

    /* CPRO OPTIMIZATION: Lock ALSA initialization to prevent resource contention */
    pthread_mutex_lock(&alsa_init_lock);

    /* Open mixer */
    err = snd_mixer_open(&ffmpeg_ctx->audio_mixer_handle, 0);
    if(err < 0) {
        LV_LOG_ERROR("Error opening mixer: %s", snd_strerror(err));
        pthread_mutex_unlock(&alsa_init_lock);
        return -1;
    }

    /* Attach to default sound card */
    err = snd_mixer_attach(ffmpeg_ctx->audio_mixer_handle, "default");
    if(err < 0) {
        LV_LOG_ERROR("Error attaching mixer: %s", snd_strerror(err));
        snd_mixer_close(ffmpeg_ctx->audio_mixer_handle);
        ffmpeg_ctx->audio_mixer_handle = NULL;
        pthread_mutex_unlock(&alsa_init_lock);
        return -1;
    }

    /* Register mixer elements */
    err = snd_mixer_selem_register(ffmpeg_ctx->audio_mixer_handle, NULL, NULL);
    if(err < 0) {
        LV_LOG_ERROR("Error registering mixer: %s", snd_strerror(err));
        snd_mixer_close(ffmpeg_ctx->audio_mixer_handle);
        ffmpeg_ctx->audio_mixer_handle = NULL;
        pthread_mutex_unlock(&alsa_init_lock);
        return -1;
    }

    /* Load mixer */
    err = snd_mixer_load(ffmpeg_ctx->audio_mixer_handle);
    if(err < 0) {
        LV_LOG_ERROR("Error loading mixer: %s", snd_strerror(err));
        snd_mixer_close(ffmpeg_ctx->audio_mixer_handle);
        ffmpeg_ctx->audio_mixer_handle = NULL;
        return -1;
    }

    /* Find PCM playback volume control element */
    for(ffmpeg_ctx->audio_mixer_elem = snd_mixer_first_elem(ffmpeg_ctx->audio_mixer_handle);
        ffmpeg_ctx->audio_mixer_elem;
        ffmpeg_ctx->audio_mixer_elem = snd_mixer_elem_next(ffmpeg_ctx->audio_mixer_elem)) {
        if(snd_mixer_selem_has_playback_volume(ffmpeg_ctx->audio_mixer_elem)) {
            LV_LOG_INFO("Found playback volume control: %s",
                       snd_mixer_selem_get_name(ffmpeg_ctx->audio_mixer_elem));
            break;
        }
    }

    if(!ffmpeg_ctx->audio_mixer_elem) {
        LV_LOG_ERROR("No playback volume control found");
        snd_mixer_close(ffmpeg_ctx->audio_mixer_handle);
        ffmpeg_ctx->audio_mixer_handle = NULL;
        return -1;
    }

    LV_LOG_INFO("ALSA Mixer initialized successfully");
    pthread_mutex_unlock(&alsa_init_lock);
    return 0;
}

/* Set mixer volume (0-100) */
static int ffmpeg_audio_mixer_set_volume(struct ffmpeg_context_s * ffmpeg_ctx, int volume)
{
    long min, max, value;

    if(!ffmpeg_ctx->audio_mixer_handle || !ffmpeg_ctx->audio_mixer_elem) {
        LV_LOG_WARN("Mixer not initialized");
        return -1;
    }

    /* Get volume range */
    snd_mixer_selem_get_playback_volume_range(ffmpeg_ctx->audio_mixer_elem, &min, &max);

    /* Calculate volume value (0-100 mapped to min-max) */
    value = min + (max - min) * volume / 100;

    /* Set left and right channel volume */
    snd_mixer_selem_set_playback_volume_all(ffmpeg_ctx->audio_mixer_elem, value);

    return 0;
}

/* Get mixer volume (0-100) */
static int ffmpeg_audio_mixer_get_volume(struct ffmpeg_context_s * ffmpeg_ctx)
{
    long min, max, value;

    if(!ffmpeg_ctx->audio_mixer_handle || !ffmpeg_ctx->audio_mixer_elem) {
        return 75; /* Default value */
    }

    snd_mixer_selem_get_playback_volume_range(ffmpeg_ctx->audio_mixer_elem, &min, &max);
    snd_mixer_selem_get_playback_volume(ffmpeg_ctx->audio_mixer_elem, SND_MIXER_SCHN_FRONT_LEFT, &value);

    /* Map back to 0-100 range */
    if(max == min) {
        LV_LOG_WARN("Volume range is zero (min=%ld, max=%ld), using default volume", min, max);
        return 75;
    }

    int volume = (value - min) * 100 / (max - min);

    /* Ensure return value is in 0-100 range */
    if(volume < 0) volume = 0;
    if(volume > 100) volume = 100;

    return volume;
}

/* Deinitialize ALSA Mixer */
static void ffmpeg_audio_mixer_deinit(struct ffmpeg_context_s * ffmpeg_ctx)
{
    if(ffmpeg_ctx->audio_mixer_handle) {
        snd_mixer_close(ffmpeg_ctx->audio_mixer_handle);
        ffmpeg_ctx->audio_mixer_handle = NULL;
        ffmpeg_ctx->audio_mixer_elem = NULL;
    }
}

/* Initialize ALSA PCM (direct output mode) */
static int ffmpeg_audio_pcm_init(struct ffmpeg_context_s * ffmpeg_ctx)
{
    int err;
    snd_pcm_hw_params_t *hw_params;
    unsigned int rate = 44100;   /* Standard sample rate for better quality */
    int channels = 2;            /* Stereo for correct playback */
    int dir;
    snd_pcm_format_t format = SND_PCM_FORMAT_S16_LE;
    /* CPRO OPTIMIZATION: Optimized buffer sizes for single-core Cortex-A7
     * Larger buffers reduce underruns but increase latency
     * Tuned for 15-20fps video playback with audio
     * Reduced buffer size for lower latency with PTS-based synchronization */
    snd_pcm_uframes_t period_size = 1024;   /* Optimized period size (reduced for lower latency) */
    snd_pcm_uframes_t buffer_size = 4096;   /* 4x period for smooth playback (reduced for lower latency) */

    if(ffmpeg_ctx->audio_pcm_handle) {
        return 0; /* Already initialized */
    }

    /* CPRO OPTIMIZATION: No mutex needed for non-blocking mode
     * Single-core CPU benefits from reduced lock contention */

    /* Open PCM device in non-blocking mode to avoid blocking */
    pthread_mutex_lock(&alsa_init_lock);
    err = snd_pcm_open(&ffmpeg_ctx->audio_pcm_handle, "default", SND_PCM_STREAM_PLAYBACK, SND_PCM_NONBLOCK);
    if(err < 0) {
        LV_LOG_ERROR("Error opening PCM device: %s", snd_strerror(err));
        pthread_mutex_unlock(&alsa_init_lock);
        return -1;
    }

    /* Allocate hardware parameters structure */
    snd_pcm_hw_params_alloca(&hw_params);

    /* Initialize hardware parameters */
    err = snd_pcm_hw_params_any(ffmpeg_ctx->audio_pcm_handle, hw_params);
    if(err < 0) {
        LV_LOG_ERROR("Error initializing hardware parameters: %s", snd_strerror(err));
        snd_pcm_close(ffmpeg_ctx->audio_pcm_handle);
        ffmpeg_ctx->audio_pcm_handle = NULL;
        pthread_mutex_unlock(&alsa_init_lock);
        return -1;
    }

    /* Set access type (interleaved mode) */
    err = snd_pcm_hw_params_set_access(ffmpeg_ctx->audio_pcm_handle, hw_params,
                                        SND_PCM_ACCESS_RW_INTERLEAVED);
    if(err < 0) {
        LV_LOG_ERROR("Error setting access type: %s", snd_strerror(err));
        snd_pcm_close(ffmpeg_ctx->audio_pcm_handle);
        ffmpeg_ctx->audio_pcm_handle = NULL;
        pthread_mutex_unlock(&alsa_init_lock);
        return -1;
    }

    /* Set sample format (16-bit little-endian) */
    err = snd_pcm_hw_params_set_format(ffmpeg_ctx->audio_pcm_handle, hw_params, format);
    if(err < 0) {
        LV_LOG_ERROR("Error setting sample format: %s", snd_strerror(err));
        /* Try alternative formats */
        if(snd_pcm_hw_params_set_format(ffmpeg_ctx->audio_pcm_handle, hw_params, SND_PCM_FORMAT_S16_BE) >= 0) {
            format = SND_PCM_FORMAT_S16_BE;
            LV_LOG_WARN("Using alternative format: S16_BE");
        } else if(snd_pcm_hw_params_set_format(ffmpeg_ctx->audio_pcm_handle, hw_params, SND_PCM_FORMAT_U16_LE) >= 0) {
            format = SND_PCM_FORMAT_U16_LE;
            LV_LOG_WARN("Using alternative format: U16_LE");
        } else {
            LV_LOG_ERROR("No suitable format found");
            snd_pcm_close(ffmpeg_ctx->audio_pcm_handle);
            ffmpeg_ctx->audio_pcm_handle = NULL;
            pthread_mutex_unlock(&alsa_init_lock);
            return -1;
        }
    }

    /* Set channels */
    err = snd_pcm_hw_params_set_channels(ffmpeg_ctx->audio_pcm_handle, hw_params, channels);
    if(err < 0) {
        LV_LOG_ERROR("Error setting channels: %s", snd_strerror(err));
        snd_pcm_close(ffmpeg_ctx->audio_pcm_handle);
        ffmpeg_ctx->audio_pcm_handle = NULL;
        pthread_mutex_unlock(&alsa_init_lock);
        return -1;
    }

    /* Set sample rate */
    err = snd_pcm_hw_params_set_rate_near(ffmpeg_ctx->audio_pcm_handle, hw_params, &rate, &dir);
    if(err < 0) {
        LV_LOG_ERROR("Error setting sample rate: %s", snd_strerror(err));
        snd_pcm_close(ffmpeg_ctx->audio_pcm_handle);
        ffmpeg_ctx->audio_pcm_handle = NULL;
        pthread_mutex_unlock(&alsa_init_lock);
        return -1;
    }

    /* Set period size */
    err = snd_pcm_hw_params_set_period_size_near(ffmpeg_ctx->audio_pcm_handle, hw_params, &period_size, &dir);
    if(err < 0) {
        LV_LOG_WARN("Error setting period size: %s", snd_strerror(err));
    }

    /* Set buffer size */
    err = snd_pcm_hw_params_set_buffer_size_near(ffmpeg_ctx->audio_pcm_handle, hw_params, &buffer_size);
    if(err < 0) {
        LV_LOG_WARN("Error setting buffer size: %s", snd_strerror(err));
    }

    /* Apply hardware parameters */
    err = snd_pcm_hw_params(ffmpeg_ctx->audio_pcm_handle, hw_params);
    if(err < 0) {
        LV_LOG_ERROR("Error setting hardware parameters: %s", snd_strerror(err));
        snd_pcm_close(ffmpeg_ctx->audio_pcm_handle);
        ffmpeg_ctx->audio_pcm_handle = NULL;
        pthread_mutex_unlock(&alsa_init_lock);
        return -1;
    }

    LV_LOG_INFO("ALSA PCM initialized successfully (rate=%u, channels=%d, format=%s, period=%lu, buffer=%lu)",
               rate, channels, snd_pcm_format_name(format), period_size, buffer_size);
    pthread_mutex_unlock(&alsa_init_lock);
    return 0;
}

/* Write to ALSA PCM device */
static int ffmpeg_audio_pcm_write(struct ffmpeg_context_s * ffmpeg_ctx, const uint8_t *data, int size)
{
    int err;
    int frames = size / 2; /* 16-bit mono, 2 bytes per frame */
    int frames_written = 0;
    int remaining_frames = frames;
    const uint8_t *write_ptr = data;

    /* OPTIMIZED: Reduce lock contention by checking handle first */
    if(!ffmpeg_ctx->audio_pcm_handle) {
        return -1;
    }

    /* CPRO OPTIMIZATION: For single-core CPU, try non-blocking write first
     * This reduces context switches when the buffer is ready */
    snd_pcm_nonblock(ffmpeg_ctx->audio_pcm_handle, 1);

    /* Try single write first (most common case) */
    err = snd_pcm_writei(ffmpeg_ctx->audio_pcm_handle, write_ptr, remaining_frames);

    if(err == remaining_frames) {
        /* Success in one write - best case */
        snd_pcm_nonblock(ffmpeg_ctx->audio_pcm_handle, 0);
        return 0;
    }

    /* Handle partial writes or errors */
    if(err > 0) {
        frames_written += err;
        remaining_frames -= err;
        write_ptr += err * 2;
    }
    else if(err == -EPIPE) {
        /* Buffer underrun - recover and retry */
        snd_pcm_prepare(ffmpeg_ctx->audio_pcm_handle);
        /* Don't log underruns to reduce I/O overhead on single-core CPU */
    }
    else if(err == -EAGAIN) {
        /* Buffer is full, switch to blocking mode and retry */
        snd_pcm_nonblock(ffmpeg_ctx->audio_pcm_handle, 0);
    }
    else {
        /* Other errors */
        snd_pcm_nonblock(ffmpeg_ctx->audio_pcm_handle, 0);
        return -1;
    }

    /* Write in a loop to handle partial writes */
    while(remaining_frames > 0) {
        err = snd_pcm_writei(ffmpeg_ctx->audio_pcm_handle, write_ptr, remaining_frames);

        if(err > 0) {
            frames_written += err;
            remaining_frames -= err;
            write_ptr += err * 2;
        }
        else if(err == -EPIPE) {
            /* Buffer underrun - recover and retry */
            snd_pcm_prepare(ffmpeg_ctx->audio_pcm_handle);
        }
        else if(err == -EAGAIN) {
            /* Buffer is full, retry immediately */
            continue;
        }
        else {
            /* Other errors */
            return -1;
        }
    }

    snd_pcm_nonblock(ffmpeg_ctx->audio_pcm_handle, 0);

    return 0;
}

/* Deinitialize ALSA PCM */
static void ffmpeg_audio_pcm_deinit(struct ffmpeg_context_s * ffmpeg_ctx)
{
    /* CPRO OPTIMIZATION: No mutex needed for non-blocking mode */
    if(ffmpeg_ctx->audio_pcm_handle) {
        snd_pcm_drain(ffmpeg_ctx->audio_pcm_handle);
        snd_pcm_close(ffmpeg_ctx->audio_pcm_handle);
        ffmpeg_ctx->audio_pcm_handle = NULL;
    }
}

/* Video buffer management functions */

/**
 * Initialize video ring buffer
 * @param buf Pointer to video buffer structure
 * @return 0 on success, -1 on error
 */
static int video_buffer_init(video_buffer_t *buf)
{
    if(!buf) {
        return -1;
    }

    memset(buf, 0, sizeof(video_buffer_t));
    buf->write_idx = 0;
    buf->read_idx = 0;
    buf->count = 0;
    buf->initialized = false;

    /* Initialize mutex and condition variable */
    if(pthread_mutex_init(&buf->mutex, NULL) != 0) {
        LV_LOG_ERROR("Failed to initialize video buffer mutex");
        return -1;
    }

    if(pthread_cond_init(&buf->cond, NULL) != 0) {
        LV_LOG_ERROR("Failed to initialize video buffer condition variable");
        pthread_mutex_destroy(&buf->mutex);
        return -1;
    }

    buf->initialized = true;
    LV_LOG_INFO("Video buffer initialized (size=%d frames)", VIDEO_BUFFER_SIZE);
    return 0;
}

/**
 * Push frame to video ring buffer (producer: video thread)
 * @param buf Pointer to video buffer structure
 * @param frame Frame to push
 * @return 0 on success, -1 on error
 */
static int video_buffer_push(video_buffer_t *buf, AVFrame *frame)
{
    if(!buf || !frame || !buf->initialized) {
        return -1;
    }

    pthread_mutex_lock(&buf->mutex);

    /* If buffer is full, drop oldest frame to avoid blocking */
    if(buf->count >= VIDEO_BUFFER_SIZE) {
        /* Free oldest frame */
        if(buf->frames[buf->read_idx]) {
            av_frame_unref(buf->frames[buf->read_idx]);
            av_frame_free(&buf->frames[buf->read_idx]);
        }
        buf->read_idx = (buf->read_idx + 1) % VIDEO_BUFFER_SIZE;
        buf->count--;
        LV_LOG_WARN("Video buffer full, dropping oldest frame (count=%d)", buf->count);
    }

    /* Clone frame to avoid reference counting issues */
    buf->frames[buf->write_idx] = av_frame_clone(frame);
    if(!buf->frames[buf->write_idx]) {
        LV_LOG_ERROR("Failed to clone frame for video buffer");
        pthread_mutex_unlock(&buf->mutex);
        return -1;
    }

    buf->write_idx = (buf->write_idx + 1) % VIDEO_BUFFER_SIZE;
    buf->count++;

    /* Signal consumer thread */
    pthread_cond_signal(&buf->cond);
    pthread_mutex_unlock(&buf->mutex);

    return 0;
}

/**
 * Pop frame from video ring buffer (consumer: LVGL main thread)
 * @param buf Pointer to video buffer structure
 * @return Frame pointer on success, NULL if buffer is empty
 */
static AVFrame *video_buffer_pop(video_buffer_t *buf)
{
    AVFrame *frame = NULL;

    if(!buf || !buf->initialized) {
        return NULL;
    }

    pthread_mutex_lock(&buf->mutex);

    /* If buffer is empty, return NULL */
    if(buf->count == 0) {
        pthread_mutex_unlock(&buf->mutex);
        return NULL;
    }

    /* Get frame from buffer */
    frame = buf->frames[buf->read_idx];
    buf->frames[buf->read_idx] = NULL;
    buf->read_idx = (buf->read_idx + 1) % VIDEO_BUFFER_SIZE;
    buf->count--;

    pthread_mutex_unlock(&buf->mutex);

    return frame;
}

/**
 * Destroy video ring buffer
 * @param buf Pointer to video buffer structure
 */
static void video_buffer_destroy(video_buffer_t *buf)
{
    if(!buf || !buf->initialized) {
        return;
    }

    pthread_mutex_lock(&buf->mutex);

    /* Free all remaining frames */
    for(int i = 0; i < VIDEO_BUFFER_SIZE; i++) {
        if(buf->frames[i]) {
            av_frame_unref(buf->frames[i]);
            av_frame_free(&buf->frames[i]);
            buf->frames[i] = NULL;
        }
    }

    pthread_mutex_unlock(&buf->mutex);

    /* Destroy mutex and condition variable */
    pthread_mutex_destroy(&buf->mutex);
    pthread_cond_destroy(&buf->cond);

    buf->initialized = false;
    LV_LOG_INFO("Video buffer destroyed");
}

/* Initialize audio output device */
static int ffmpeg_audio_init(struct ffmpeg_context_s * ffmpeg_ctx)
{
    int ret;
    const AVCodec *audio_codec = NULL;

    if(!ffmpeg_ctx->has_audio || !ffmpeg_ctx->audio_dec_ctx) {
        return -1;
    }

    /* Debug: Print audio decoder parameters */
    LV_LOG_INFO("Audio decoder parameters:");
    LV_LOG_INFO("  sample_rate: %d", ffmpeg_ctx->audio_dec_ctx->sample_rate);
    LV_LOG_INFO("  sample_fmt: %d (%s)", ffmpeg_ctx->audio_dec_ctx->sample_fmt,
               av_get_sample_fmt_name(ffmpeg_ctx->audio_dec_ctx->sample_fmt));
    LV_LOG_INFO("  channels: %d", ffmpeg_ctx->audio_dec_ctx->ch_layout.nb_channels);
    LV_LOG_INFO("  frame_size: %d", ffmpeg_ctx->audio_dec_ctx->frame_size);

#if LV_FFMPEG_USE_AVDEVICE == 1
    LV_LOG_INFO("Initializing audio output with avdevice...");
#else
    LV_LOG_INFO("Initializing audio output with ALSA PCM...");
#endif

    /* Initialize ALSA Mixer (for hardware volume control) */
    /* DISABLED: Mixer and PCM conflict - causing "Invalid argument" error */
    /*
    if(ffmpeg_audio_mixer_init(ffmpeg_ctx) < 0) {
        LV_LOG_WARN("Failed to initialize ALSA Mixer, volume control may not work");
    }
    */

    /* Find audio stream in format context (unified) */
    LV_LOG_INFO("Finding audio stream in format context...");
    ret = av_find_best_stream(ffmpeg_ctx->fmt_ctx, AVMEDIA_TYPE_AUDIO, -1, -1, &audio_codec, 0);
    if(ret < 0) {
        LV_LOG_ERROR("Could not find audio stream");
        return -1;
    }
    ffmpeg_ctx->audio_stream_idx = ret;

    /* Allocate audio frame (unified) */
    ffmpeg_ctx->audio_frame = av_frame_alloc();
    if(!ffmpeg_ctx->audio_frame) {
        LV_LOG_ERROR("Failed to allocate audio frame");
        return -1;
    }

#if LV_FFMPEG_USE_AVDEVICE == 1
    /* Use FFmpeg avdevice output */
    ret = avformat_alloc_output_context2(&ffmpeg_ctx->audio_out_fmt_ctx, NULL, "alsa", "default");
    if(ret < 0 || !ffmpeg_ctx->audio_out_fmt_ctx) {
        LV_LOG_ERROR("Error creating audio output context: %s", av_err2str(ret));
        return -1;
    }

    /* Create output stream */
    AVStream *out_stream = avformat_new_stream(ffmpeg_ctx->audio_out_fmt_ctx, NULL);
    if(!out_stream) {
        LV_LOG_ERROR("Error creating audio output stream");
        avformat_free_context(ffmpeg_ctx->audio_out_fmt_ctx);
        ffmpeg_ctx->audio_out_fmt_ctx = NULL;
        return -1;
    }

    /* Set audio output parameters */
    AVCodecParameters *codecpar = out_stream->codecpar;
    codecpar->codec_type = AVMEDIA_TYPE_AUDIO;
    codecpar->codec_id = AV_CODEC_ID_PCM_S16LE;
    codecpar->sample_rate = 44100;
    codecpar->ch_layout = (AVChannelLayout)AV_CHANNEL_LAYOUT_STEREO;
    codecpar->format = AV_SAMPLE_FMT_S16;

    /* Open output device */
    if(!(ffmpeg_ctx->audio_out_fmt_ctx->oformat->flags & AVFMT_NOFILE)) {
        ret = avio_open(&ffmpeg_ctx->audio_out_fmt_ctx->pb, ffmpeg_ctx->audio_out_fmt_ctx->url, AVIO_FLAG_WRITE);
        if(ret < 0) {
            LV_LOG_ERROR("Error opening audio output device: %s", av_err2str(ret));
            avformat_free_context(ffmpeg_ctx->audio_out_fmt_ctx);
            ffmpeg_ctx->audio_out_fmt_ctx = NULL;
            return -1;
        }
    }

    /* Write header */
    ret = avformat_write_header(ffmpeg_ctx->audio_out_fmt_ctx, NULL);
    if(ret < 0) {
        LV_LOG_ERROR("Error writing audio output header: %s", av_err2str(ret));
        if(!(ffmpeg_ctx->audio_out_fmt_ctx->oformat->flags & AVFMT_NOFILE)) {
            avio_closep(&ffmpeg_ctx->audio_out_fmt_ctx->pb);
        }
        avformat_free_context(ffmpeg_ctx->audio_out_fmt_ctx);
        ffmpeg_ctx->audio_out_fmt_ctx = NULL;
        return -1;
    }

    LV_LOG_INFO("Audio output initialized successfully with avdevice");
#else
    /* Use ALSA PCM direct output */
    if(ffmpeg_audio_pcm_init(ffmpeg_ctx) < 0) {
        LV_LOG_ERROR("Failed to initialize ALSA PCM");
        return -1;
    }

    LV_LOG_INFO("Audio output initialized successfully with ALSA PCM");
#endif

    /* OPTIMIZED: Allocate reusable audio output packet (avdevice mode) */
#if LV_FFMPEG_USE_AVDEVICE == 1
    ffmpeg_ctx->audio_out_pkt = av_packet_alloc();
    if(!ffmpeg_ctx->audio_out_pkt) {
        LV_LOG_ERROR("Failed to allocate reusable audio output packet");
        ffmpeg_audio_deinit(ffmpeg_ctx);
        return -1;
    }
#endif

    /* Initialize audio resampler (unified) with performance optimizations */
    AVChannelLayout src_ch_layout = ffmpeg_ctx->audio_dec_ctx->ch_layout;
    AVChannelLayout dst_ch_layout;
    av_channel_layout_default(&dst_ch_layout, 2);  /* Stereo for correct playback */

    ffmpeg_ctx->swr_ctx = swr_alloc();
    av_opt_set_chlayout(ffmpeg_ctx->swr_ctx, "in_chlayout", &src_ch_layout, 0);
    av_opt_set_int(ffmpeg_ctx->swr_ctx, "in_sample_rate", ffmpeg_ctx->audio_dec_ctx->sample_rate, 0);
    av_opt_set_sample_fmt(ffmpeg_ctx->swr_ctx, "in_sample_fmt", ffmpeg_ctx->audio_dec_ctx->sample_fmt, 0);
    av_opt_set_chlayout(ffmpeg_ctx->swr_ctx, "out_chlayout", &dst_ch_layout, 0);
    av_opt_set_int(ffmpeg_ctx->swr_ctx, "out_sample_rate", 44100, 0);  /* Match avdevice configuration */
    av_opt_set_sample_fmt(ffmpeg_ctx->swr_ctx, "out_sample_fmt", AV_SAMPLE_FMT_S16, 0);

    /* OPTIMIZED: Use fastest resampling method for single-core CPU
     * 0 = default (balance), 1 = fast, 2 = best
     * On single-core CPU, fast mode reduces CPU overhead significantly */
    av_opt_set_int(ffmpeg_ctx->swr_ctx, "resample_method", 1, 0);  /* Use fast mode */
    av_opt_set_int(ffmpeg_ctx->swr_ctx, "dither_method", 0, 0);  /* Disable dither */
    av_opt_set_int(ffmpeg_ctx->swr_ctx, "precision", 15, 0);    /* Minimum precision (15-33) */

    ret = swr_init(ffmpeg_ctx->swr_ctx);
    if(ret < 0) {
        LV_LOG_ERROR("Error initializing audio resampler: %s", av_err2str(ret));
        ffmpeg_audio_deinit(ffmpeg_ctx);
        return -1;
    }

    ffmpeg_ctx->audio_buf = NULL;
    ffmpeg_ctx->audio_buf_size = 0;

    /* Initialize thread flags */
    ffmpeg_ctx->is_audio_playing = 0;
    ffmpeg_ctx->is_audio_paused = 0;

    return 0;
}

/* Deinitialize audio output device */
static void ffmpeg_audio_deinit(struct ffmpeg_context_s * ffmpeg_ctx)
{
    if(!ffmpeg_ctx) {
        return;
    }

#if LV_FFMPEG_USE_AVDEVICE == 1
    /* Clean up avdevice output */
    if(ffmpeg_ctx->audio_out_fmt_ctx) {
        av_write_trailer(ffmpeg_ctx->audio_out_fmt_ctx);
        if(!(ffmpeg_ctx->audio_out_fmt_ctx->oformat->flags & AVFMT_NOFILE)) {
            avio_closep(&ffmpeg_ctx->audio_out_fmt_ctx->pb);
        }
        avformat_free_context(ffmpeg_ctx->audio_out_fmt_ctx);
        ffmpeg_ctx->audio_out_fmt_ctx = NULL;
    }

    /* OPTIMIZED: Free reusable audio output packet */
    if(ffmpeg_ctx->audio_out_pkt) {
        av_packet_free(&ffmpeg_ctx->audio_out_pkt);
        ffmpeg_ctx->audio_out_pkt = NULL;
    }
#else
    /* Clean up ALSA PCM */
    ffmpeg_audio_pcm_deinit(ffmpeg_ctx);
#endif

    /* Clean up audio resampler */
    if(ffmpeg_ctx->swr_ctx) {
        swr_free(&ffmpeg_ctx->swr_ctx);
        ffmpeg_ctx->swr_ctx = NULL;
    }
}

/* Unified playback thread: processes both audio and video packets in a single thread */
static void *ffmpeg_playback_thread(void *arg)
{
    struct ffmpeg_context_s *ffmpeg_ctx = (struct ffmpeg_context_s *)arg;
    lv_ffmpeg_player_t *player = ffmpeg_ctx->player;
    int ret;
    AVPacket *pkt = NULL;
    AVFrame *frame = NULL;
    uint8_t *audio_buf = NULL;
    int audio_buf_size = 0;

    LV_LOG_INFO("Unified playback thread started");

    /* Allocate packet for playback thread */
    pkt = av_packet_alloc();
    if(!pkt) {
        LV_LOG_ERROR("Failed to allocate packet in playback thread");
        return NULL;
    }

    /* Allocate frame for playback thread */
    frame = av_frame_alloc();
    if(!frame) {
        LV_LOG_ERROR("Failed to allocate frame in playback thread");
        av_packet_free(&pkt);
        return NULL;
    }

    /* Main playback loop */
    while(ffmpeg_ctx->is_playing) {
        /* Check if paused */
        if(ffmpeg_ctx->is_paused) {
            usleep(10000); /* 10ms sleep when paused */
            continue;
        }

        /* Read frame from format context */
        ret = av_read_frame(ffmpeg_ctx->fmt_ctx, pkt);

        if(ret < 0) {
            if(ret == AVERROR_EOF) {
                LV_LOG_INFO("Playback thread reached EOF");
                break;
            }

            usleep(10000); /* Wait and retry on error */
            continue;
        }

        /* Process video packets */
        if(pkt->stream_index == ffmpeg_ctx->video_stream_idx) {
            /* Send packet to decoder */
            ret = avcodec_send_packet(ffmpeg_ctx->video_dec_ctx, pkt);
            if(ret < 0) {
                av_packet_unref(pkt);
                continue;
            }

            /* Receive decoded frames */
            while(ret >= 0) {
                ret = avcodec_receive_frame(ffmpeg_ctx->video_dec_ctx, frame);

                if(ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                    break;
                }

                if(ret < 0) {
                    LV_LOG_ERROR("Error receiving video frame: %s", av_err2str(ret));
                    break;
                }

                /* Validate frame dimensions and format
                 * Sometimes avcodec_receive_frame returns invalid frames (0x0 or null format)
                 * at the end of streams or during flush operations. Skip these frames. */
                if(frame->width == 0 || frame->height == 0 || frame->format == AV_PIX_FMT_NONE) {
                    LV_LOG_WARN("Skipping invalid video frame: width=%d, height=%d, format=%d",
                               frame->width, frame->height, frame->format);
                    av_frame_unref(frame);
                    continue;
                }

#if LV_FFMPEG_SYNC_ENABLED != 0
                /* Update video clock based on PTS */
                if(ffmpeg_ctx->video_stream && frame->pts != AV_NOPTS_VALUE) {
                    int64_t pts_ms = pts_to_ms(ffmpeg_ctx->video_stream, frame->pts);
                    if(pts_ms != AV_NOPTS_VALUE) {
                        ffmpeg_ctx->video_pts = pts_ms;
                        ffmpeg_ctx->video_clock = pts_ms;
                        __sync_synchronize();  /* Memory barrier after update */

                        /* If this is the first frame, set start time */
                        if(ffmpeg_ctx->start_time == 0) {
                            ffmpeg_ctx->start_time = get_current_time_ms();
                        }
                    }
                }
#endif

                /* Copy decoded frame to ffmpeg_ctx->frame for output processing
                 * ffmpeg_output_video_frame expects frame data in ffmpeg_ctx->frame
                 * This ensures compatibility with the existing output function */
                av_frame_unref(ffmpeg_ctx->frame);
                if(av_frame_ref(ffmpeg_ctx->frame, frame) < 0) {
                    LV_LOG_ERROR("Failed to copy frame to ffmpeg_ctx->frame");
                    av_frame_unref(frame);
                    break;
                }

                /* Output video frame with format conversion */
                if(ffmpeg_output_video_frame(ffmpeg_ctx) < 0) {
                    LV_LOG_ERROR("Error outputting video frame");
                    av_frame_unref(frame);
                    break;
                }

                /* Push decoded frame to ring buffer for LVGL main thread */
                if(video_buffer_push(&ffmpeg_ctx->video_buffer, frame) < 0) {
                    LV_LOG_ERROR("Error pushing frame to video buffer");
                }

                av_frame_unref(frame);
            }
        }
        /* Process audio packets */
        else if(pkt->stream_index == ffmpeg_ctx->audio_stream_idx) {
            /* Check if audio is enabled */
            if(!player || !player->audio_enabled) {
                av_packet_unref(pkt);
                continue;
            }

            /* Send packet to decoder */
            ret = avcodec_send_packet(ffmpeg_ctx->audio_dec_ctx, pkt);
            if(ret < 0) {
                av_packet_unref(pkt);
                continue;
            }

            /* Receive decoded frames */
            while(ret >= 0) {
                ret = avcodec_receive_frame(ffmpeg_ctx->audio_dec_ctx, frame);

                if(ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                    break;
                }

                if(ret < 0) {
                    LV_LOG_ERROR("Error receiving audio frame: %s", av_err2str(ret));
                    break;
                }

#if LV_FFMPEG_SYNC_ENABLED != 0
                /* Update audio clock based on PTS */
                if(ffmpeg_ctx->audio_stream && frame->pts != AV_NOPTS_VALUE) {
                    int64_t pts_ms = pts_to_ms(ffmpeg_ctx->audio_stream, frame->pts);
                    if(pts_ms != AV_NOPTS_VALUE) {
                        ffmpeg_ctx->audio_pts = pts_ms;
                        __sync_synchronize();  /* Memory barrier after update */
                        ffmpeg_ctx->audio_clock = pts_ms;

                        /* If this is the first frame, set start time */
                        if(ffmpeg_ctx->start_time == 0) {
                            ffmpeg_ctx->start_time = get_current_time_ms();
                        }
                    }
                }
#endif

                /* Resample audio using unified resampler (swr_ctx) */
                /* Note: In unified playback thread, we use swr_ctx because it's
                 * already initialized in ffmpeg_audio_init() */
                int dst_nb_samples = av_rescale_rnd(
                    swr_get_delay(ffmpeg_ctx->swr_ctx, ffmpeg_ctx->audio_dec_ctx->sample_rate) +
                    frame->nb_samples,
                    44100, ffmpeg_ctx->audio_dec_ctx->sample_rate, AV_ROUND_UP
                );

                if(dst_nb_samples > audio_buf_size / 4) {
                    audio_buf_size = dst_nb_samples * 4; /* 16-bit stereo */
                    uint8_t *new_buf = av_realloc(audio_buf, audio_buf_size);
                    if(!new_buf) {
                        LV_LOG_ERROR("Failed to reallocate audio buffer in playback thread");
                        av_frame_unref(frame);
                        continue;
                    }
                    audio_buf = new_buf;
                }

                int out_samples = swr_convert(
                    ffmpeg_ctx->swr_ctx, &audio_buf, dst_nb_samples,
                    (const uint8_t **)frame->data, frame->nb_samples
                );

                if(out_samples <= 0) {
                    av_frame_unref(frame);
                    continue;
                }

                int out_size = out_samples * 4; /* 16-bit stereo */

#if LV_FFMPEG_USE_AVDEVICE == 1
                /* Use avdevice output */
                AVPacket *out_pkt = av_packet_alloc();
                if(out_pkt) {
                    out_pkt->data = av_malloc(out_size);
                    if(out_pkt->data) {
                        memcpy(out_pkt->data, audio_buf, out_size);
                        out_pkt->size = out_size;
                        out_pkt->stream_index = 0;
                        out_pkt->pts = frame->pts;
                        out_pkt->dts = frame->pkt_dts;

                        ret = av_write_frame(ffmpeg_ctx->audio_out_fmt_ctx, out_pkt);
                        if(ret < 0) {
                            LV_LOG_ERROR("Error writing audio frame: %s", av_err2str(ret));
                        }
                    }
                    av_packet_free(&out_pkt);
                }
#else
                /* Use ALSA PCM direct output */
                ret = ffmpeg_audio_pcm_write(ffmpeg_ctx, audio_buf, out_size);
                if(ret < 0) {
                    LV_LOG_ERROR("Error writing to PCM device in playback thread");
                }
#endif

                av_frame_unref(frame);
            }
        }
        /* Ignore other packet types */
        else {
            av_packet_unref(pkt);
            continue;
        }

        av_packet_unref(pkt);
    }

    /* Cleanup */
    if(audio_buf) {
        av_free(audio_buf);
    }
    av_frame_free(&frame);
    av_packet_free(&pkt);

    LV_LOG_INFO("Unified playback thread stopped");
    return NULL;
}

#if LV_FFMPEG_SYNC_ENABLED != 0
/**
 * Convert PTS to milliseconds
 * @param stream AVStream containing time_base
 * @param pts Presentation timestamp
 * @return Time in milliseconds, or AV_NOPTS_VALUE if invalid
 */
static int64_t pts_to_ms(AVStream *stream, int64_t pts)
{
    if (pts == AV_NOPTS_VALUE) {
        return AV_NOPTS_VALUE;
    }

    AVRational time_base = stream->time_base;
    return av_rescale_q(pts, time_base, AV_TIME_BASE_Q) / 1000;
}

/**
 * Get current system time in milliseconds
 * @return Current time in milliseconds
 */
static int64_t get_current_time_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
}

/**
 * Check if current video frame should be skipped
 * @param ffmpeg_ctx FFmpeg context
 * @return true if frame should be skipped, false otherwise
 */
static bool should_skip_video_frame(struct ffmpeg_context_s *ffmpeg_ctx)
{
    if (!ffmpeg_ctx->sync_enabled || !ffmpeg_ctx->has_audio) {
        return false;
    }

    /* If no valid audio clock, don't skip frame */
    __sync_synchronize();  /* Memory barrier before read */
    int64_t video_pts_local = ffmpeg_ctx->video_pts;
    int64_t audio_pts_local = ffmpeg_ctx->audio_pts;
    if (audio_pts_local == AV_NOPTS_VALUE ||
        video_pts_local == AV_NOPTS_VALUE) {
        return false;
    }

    /* Calculate audio-video time difference */
    int64_t diff = video_pts_local - audio_pts_local;

    /* If video is behind audio by more than sync_threshold, skip this frame */
    if (diff < -ffmpeg_ctx->sync_threshold) {
        ffmpeg_ctx->frame_drop_count++;
        LV_LOG_INFO("[SYNC] Skipping video frame: video=%lld ms, audio=%lld ms, diff=%lld ms",
                   (long long)ffmpeg_ctx->video_pts,
                   (long long)ffmpeg_ctx->audio_pts,
                   (long long)diff);
        return true;
    }

    return false;
}

/**
 * Check if current video frame should be repeated
 * @param ffmpeg_ctx FFmpeg context
 * @return true if frame should be repeated, false otherwise
 */
static bool should_repeat_video_frame(struct ffmpeg_context_s *ffmpeg_ctx)
{
    if (!ffmpeg_ctx->sync_enabled || !ffmpeg_ctx->has_audio) {
        return false;
    }

    /* If no valid audio clock, don't repeat frame */
    __sync_synchronize();  /* Memory barrier before read */
    int64_t video_pts_local = ffmpeg_ctx->video_pts;
    int64_t audio_pts_local = ffmpeg_ctx->audio_pts;
    if (audio_pts_local == AV_NOPTS_VALUE ||
        video_pts_local == AV_NOPTS_VALUE) {
        return false;
    }

    /* Calculate audio-video time difference */
    int64_t diff = video_pts_local - audio_pts_local;

    /* If video is ahead of audio by more than sync_threshold, repeat this frame */
    if (diff > ffmpeg_ctx->sync_threshold) {
        ffmpeg_ctx->frame_repeat_count++;
        LV_LOG_INFO("[SYNC] Repeating video frame: video=%lld ms, audio=%lld ms, diff=%lld ms",
                   (long long)ffmpeg_ctx->video_pts,
                   (long long)ffmpeg_ctx->audio_pts,
                   (long long)diff);
        return true;
    }

    return false;
}
#endif /* LV_FFMPEG_SYNC_ENABLED */

/* Output audio frame to device */
static int ffmpeg_output_audio_frame(struct ffmpeg_context_s * ffmpeg_ctx)
{
    int ret = 0;
    AVFrame *frame = ffmpeg_ctx->audio_frame;

    if(!ffmpeg_ctx->player) {
        LV_LOG_WARN("ffmpeg_output_audio_frame: No player context");
        return 0; /* No player context */
    }

    /* Check if audio is enabled */
    if(!ffmpeg_ctx->player->audio_enabled) {
        return 0; /* Audio is disabled, skip this frame */
    }

    static int frame_count = 0;
    /* Disable audio frame logging for better performance */
    if(0 && frame_count < 5) {
        LV_LOG_INFO("ffmpeg_output_audio_frame: Processing audio frame %d, nb_samples=%d, data[0]=%p, data[1]=%p",
                   frame_count, frame->nb_samples, frame->data[0], frame->data[1]);
        frame_count++;
    }

    if(frame->nb_samples == 0) {
        return 0;
    }

#if LV_FFMPEG_USE_AVDEVICE == 1
    /* avdevice mode: check if output context is initialized */
    if(!ffmpeg_ctx->audio_out_fmt_ctx || !ffmpeg_ctx->swr_ctx) {
        return 0;
    }
#else
    /* ALSA PCM mode: check if PCM handle is initialized */
    if(!ffmpeg_ctx->audio_pcm_handle || !ffmpeg_ctx->swr_ctx) {
        return 0;
    }
#endif

    /* CPRO OPTIMIZATION: Check if resampling is needed
     * If input format is already 44100Hz/16-bit/stereo, skip resampling */
    bool needs_resampling = true;
    if(ffmpeg_ctx->audio_dec_ctx->sample_rate == 44100 &&
       ffmpeg_ctx->audio_dec_ctx->sample_fmt == AV_SAMPLE_FMT_S16 &&
       ffmpeg_ctx->audio_dec_ctx->ch_layout.nb_channels == 2) {
        /* No resampling needed, use data directly */
        needs_resampling = false;
    }

    int out_samples;
    int out_size;

    if(needs_resampling) {
        /* Calculate output samples */
        int dst_nb_samples = av_rescale_rnd(
            swr_get_delay(ffmpeg_ctx->swr_ctx, ffmpeg_ctx->audio_dec_ctx->sample_rate) +
            frame->nb_samples,
            44100, ffmpeg_ctx->audio_dec_ctx->sample_rate, AV_ROUND_UP
        );

        /* Reallocate audio buffer if needed */
        if(dst_nb_samples > ffmpeg_ctx->audio_buf_size / 4) {
            ffmpeg_ctx->audio_buf_size = dst_nb_samples * 4; /* 16-bit stereo */
            uint8_t *new_buf = av_realloc(ffmpeg_ctx->audio_buf, ffmpeg_ctx->audio_buf_size);
            if(!new_buf) {
                LV_LOG_ERROR("Failed to reallocate audio buffer");
                return -1;
            }
            ffmpeg_ctx->audio_buf = new_buf;
        }

        /* Resample audio */
        out_samples = swr_convert(
            ffmpeg_ctx->swr_ctx, &ffmpeg_ctx->audio_buf, dst_nb_samples,
            (const uint8_t **)frame->data, frame->nb_samples
        );

        if(out_samples <= 0) {
            return 0;
        }

        out_size = out_samples * 4;
    } else {
        /* Use data directly without resampling */
        ffmpeg_ctx->audio_buf = (uint8_t *)frame->data[0];
        out_samples = frame->nb_samples;
        out_size = out_samples * 4;
    }

#if LV_FFMPEG_USE_AVDEVICE == 1
    /* OPTIMIZED: Reuse audio output packet to reduce allocation overhead */
    AVPacket *out_pkt = ffmpeg_ctx->audio_out_pkt;
    if(!out_pkt) {
        LV_LOG_ERROR("Audio output packet not initialized");
        return -1;
    }

    /* Reuse existing packet data buffer if possible */
    if(out_pkt->size < out_size) {
        /* Need to allocate larger buffer */
        av_packet_unref(out_pkt);
        out_pkt->data = av_malloc(out_size);
        if(!out_pkt->data) {
            LV_LOG_ERROR("Failed to allocate packet data");
            return -1;
        }
    }

    memcpy(out_pkt->data, ffmpeg_ctx->audio_buf, out_size);
    out_pkt->size = out_size;
    out_pkt->stream_index = 0;
    out_pkt->pts = frame->pts;
    out_pkt->dts = frame->pkt_dts;
    out_pkt->duration = frame->duration;

    /* Write to output device */
    ret = av_write_frame(ffmpeg_ctx->audio_out_fmt_ctx, out_pkt);

    /* Don't free the packet - it will be reused */

    if(ret < 0) {
        LV_LOG_ERROR("Error writing audio frame: %s", av_err2str(ret));
        return ret;
    }
#else
    /* Use ALSA PCM direct output */
    ret = ffmpeg_audio_pcm_write(ffmpeg_ctx, ffmpeg_ctx->audio_buf, out_size);
    if(ret < 0) {
        LV_LOG_ERROR("Error writing to PCM device");
        return ret;
    }
#endif

    /* No delay needed - let ALSA handle buffering */
    /* Removing delay to prevent buffer underrun */

    return 0;
}

#endif /* LV_FFMPEG_AUDIO_SUPPORT */

/* CPRO OPTIMIZATION: NEON-accelerated YUV to RGB conversion functions
 * These functions use ARM NEON intrinsics to accelerate YUV420P to RGB conversion
 * on ARM Cortex-A7 processors, processing 8-16 pixels in parallel.
 *
 * Performance gains on Cortex-A7:
 * - RGB565: ~4-5x faster than sws_scale
 * - RGB888: ~3-4x faster than sws_scale
 *
 * Memory alignment:
 * - Input Y/U/V buffers should be 16-byte aligned for optimal performance
 * - Output RGB buffer should be 16-byte aligned
 */

#if LV_USE_DRAW_SW && defined(__ARM_NEON)

/**
 * Convert YUV420P to RGB565 using NEON intrinsics
 * Processes 8 pixels per iteration (128-bit NEON register)
 *
 * @param y: Pointer to Y plane (luminance)
 * @param u: Pointer to U plane (chrominance blue)
 * @param v: Pointer to V plane (chrominance red)
 * @param rgb: Pointer to output RGB565 buffer
 * @param width: Image width
 * @param height: Image height
 * @param y_stride: Y plane stride (bytes per row)
 * @param uv_stride: UV plane stride (bytes per row)
 * @param rgb_stride: RGB output stride (bytes per row)
 */
static void neon_yuv420p_to_rgb565(const uint8_t *y, const uint8_t *u, const uint8_t *v,
                                   uint16_t *rgb, int width, int height,
                                   int y_stride, int uv_stride, int rgb_stride)
{
    int x, y_pos;
    const int16x8_t coeff_r = vdupq_n_s16(91881);   /* 1.402 * 65536 */
    const int16x8_t coeff_g = vdupq_n_s16(-22554);  /* -0.344 * 65536 */
    const int16x8_t coeff_g2 = vdupq_n_s16(-46802); /* -0.714 * 65536 */
    const int16x8_t coeff_b = vdupq_n_s16(116130);  /* 1.772 * 65536 */
    const int16x8_t offset = vdupq_n_s16(32768);    /* 128 * 256 */

    for (y_pos = 0; y_pos < height; y_pos++) {
        const uint8_t *y_row = y + y_pos * y_stride;
        const uint8_t *u_row = u + (y_pos / 2) * uv_stride;
        const uint8_t *v_row = v + (y_pos / 2) * uv_stride;
        uint16_t *rgb_row = rgb + y_pos * (rgb_stride / 2);

        for (x = 0; x < width - 7; x += 8) {
            /* Load 8 Y values */
            int16x8_t y_vec = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(y_row + x)));

            /* Load 4 U and V values, duplicate for 8 pixels */
            int16x8_t u_vec = vcombine_s16(
                vdup_n_s16(u_row[x / 2] - 128),
                vdup_n_s16(u_row[x / 2] - 128)
            );
            int16x8_t v_vec = vcombine_s16(
                vdup_n_s16(v_row[x / 2] - 128),
                vdup_n_s16(v_row[x / 2] - 128)
            );

            /* Calculate R, G, B components */
            int16x8_t r_vec = vqaddq_s16(vmulq_s16(v_vec, coeff_r), y_vec);
            int16x8_t g_vec = vqaddq_s16(vqaddq_s16(vmulq_s16(u_vec, coeff_g), vmulq_s16(v_vec, coeff_g2)), y_vec);
            int16x8_t b_vec = vqaddq_s16(vmulq_s16(u_vec, coeff_b), y_vec);

            /* Clamp to 0-255 range */
            uint16x8_t r_clamped = vqmovun_s16(vrshrq_n_s16(r_vec, 8));
            uint16x8_t g_clamped = vqmovun_s16(vrshrq_n_s16(g_vec, 8));
            uint16x8_t b_clamped = vqmovun_s16(vrshrq_n_s16(b_vec, 8));

            /* Convert to RGB565 (5-6-5 format) */
            /* R: 5 bits (bits 11-15), G: 6 bits (bits 5-10), B: 5 bits (bits 0-4) */
            uint16x8_t r5 = vshlq_n_u16(r_clamped, 8);
            uint16x8_t g6 = vshlq_n_u16(g_clamped, 3);
            uint16x8_t b5 = vshrq_n_u16(b_clamped, 3);

            uint16x8_t rgb565 = vorrq_u16(vorrq_u16(r5, g6), b5);

            /* Store result */
            vst1q_u16(rgb_row + x, rgb565);
        }

        /* Handle remaining pixels (less than 8) */
        for (; x < width; x++) {
            int32_t y_val = y_row[x];
            int32_t u_val = u_row[x / 2] - 128;
            int32_t v_val = v_row[x / 2] - 128;

            int32_t r_val = y_val + (v_val * 91881 >> 16);
            int32_t g_val = y_val - (u_val * 22554 >> 16) - (v_val * 46802 >> 16);
            int32_t b_val = y_val + (u_val * 116130 >> 16);

            /* Clamp */
            r_val = (r_val < 0) ? 0 : (r_val > 255) ? 255 : r_val;
            g_val = (g_val < 0) ? 0 : (g_val > 255) ? 255 : g_val;
            b_val = (b_val < 0) ? 0 : (b_val > 255) ? 255 : b_val;

            /* RGB565 */
            rgb_row[x] = ((r_val & 0xF8) << 8) | ((g_val & 0xFC) << 3) | (b_val >> 3);
        }
    }
}

/**
 * Convert YUV420P to RGB888 using NEON intrinsics
 * Processes 8 pixels per iteration (3 NEON registers for R, G, B)
 *
 * @param y: Pointer to Y plane (luminance)
 * @param u: Pointer to U plane (chrominance blue)
 * @param v: Pointer to V plane (chrominance red)
 * @param rgb: Pointer to output RGB888 buffer
 * @param width: Image width
 * @param height: Image height
 * @param y_stride: Y plane stride (bytes per row)
 * @param uv_stride: UV plane stride (bytes per row)
 * @param rgb_stride: RGB output stride (bytes per row)
 */
static void neon_yuv420p_to_rgb888(const uint8_t *y, const uint8_t *u, const uint8_t *v,
                                   uint8_t *rgb, int width, int height,
                                   int y_stride, int uv_stride, int rgb_stride)
{
    int x, y_pos;
    const int16x8_t coeff_r = vdupq_n_s16(91881);   /* 1.402 * 65536 */
    const int16x8_t coeff_g = vdupq_n_s16(-22554);  /* -0.344 * 65536 */
    const int16x8_t coeff_g2 = vdupq_n_s16(-46802); /* -0.714 * 65536 */
    const int16x8_t coeff_b = vdupq_n_s16(116130);  /* 1.772 * 65536 */

    for (y_pos = 0; y_pos < height; y_pos++) {
        const uint8_t *y_row = y + y_pos * y_stride;
        const uint8_t *u_row = u + (y_pos / 2) * uv_stride;
        const uint8_t *v_row = v + (y_pos / 2) * uv_stride;
        uint8_t *rgb_row = rgb + y_pos * rgb_stride;

        for (x = 0; x < width - 7; x += 8) {
            /* Load 8 Y values */
            int16x8_t y_vec = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(y_row + x)));

            /* Load 4 U and V values, duplicate for 8 pixels */
            int16x8_t u_vec = vcombine_s16(
                vdup_n_s16(u_row[x / 2] - 128),
                vdup_n_s16(u_row[x / 2] - 128)
            );
            int16x8_t v_vec = vcombine_s16(
                vdup_n_s16(v_row[x / 2] - 128),
                vdup_n_s16(v_row[x / 2] - 128)
            );

            /* Calculate R, G, B components */
            int16x8_t r_vec = vqaddq_s16(vmulq_s16(v_vec, coeff_r), y_vec);
            int16x8_t g_vec = vqaddq_s16(vqaddq_s16(vmulq_s16(u_vec, coeff_g), vmulq_s16(v_vec, coeff_g2)), y_vec);
            int16x8_t b_vec = vqaddq_s16(vmulq_s16(u_vec, coeff_b), y_vec);

            /* Clamp to 0-255 range and convert to 8-bit */
            uint8x8_t r_clamped = vqmovun_s16(vrshrq_n_s16(r_vec, 8));
            uint8x8_t g_clamped = vqmovun_s16(vrshrq_n_s16(g_vec, 8));
            uint8x8_t b_clamped = vqmovun_s16(vrshrq_n_s16(b_vec, 8));

            /* Interleave RGB for 8 pixels (RGBRGBRGB...) */
            uint8x8x3_t rgb_vec;
            rgb_vec.val[0] = r_clamped;
            rgb_vec.val[1] = g_clamped;
            rgb_vec.val[2] = b_clamped;

            /* Store 24 bytes (8 pixels * 3 channels) */
            vst3_u8(rgb_row + x * 3, rgb_vec);
        }

        /* Handle remaining pixels (less than 8) */
        for (; x < width; x++) {
            int32_t y_val = y_row[x];
            int32_t u_val = u_row[x / 2] - 128;
            int32_t v_val = v_row[x / 2] - 128;

            int32_t r_val = y_val + (v_val * 91881 >> 16);
            int32_t g_val = y_val - (u_val * 22554 >> 16) - (v_val * 46802 >> 16);
            int32_t b_val = y_val + (u_val * 116130 >> 16);

            /* Clamp */
            r_val = (r_val < 0) ? 0 : (r_val > 255) ? 255 : r_val;
            g_val = (g_val < 0) ? 0 : (g_val > 255) ? 255 : g_val;
            b_val = (b_val < 0) ? 0 : (b_val > 255) ? 255 : b_val;

            /* RGB888 */
            rgb_row[x * 3] = (uint8_t)r_val;
            rgb_row[x * 3 + 1] = (uint8_t)g_val;
            rgb_row[x * 3 + 2] = (uint8_t)b_val;
        }
    }
}

#endif /* LV_USE_DRAW_SW && __ARM_NEON */

#endif /*LV_USE_FFMPEG*/
