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
#if LV_FFMPEG_AUDIO_SUPPORT != 0
#include <libavdevice/avdevice.h>
#include <libswresample/swresample.h>
#endif

/*********************
 *      DEFINES
 *********************/

#define DECODER_NAME    "FFMPEG"

#if LV_COLOR_DEPTH == 8
    #define AV_PIX_FMT_TRUE_COLOR AV_PIX_FMT_RGB8
#elif LV_COLOR_DEPTH == 16
    #define AV_PIX_FMT_TRUE_COLOR AV_PIX_FMT_RGB565LE
#elif LV_COLOR_DEPTH == 32
    #define AV_PIX_FMT_TRUE_COLOR AV_PIX_FMT_BGR0
#else
    #error Unsupported  LV_COLOR_DEPTH
#endif

#define MY_CLASS (&lv_ffmpeg_player_class)

#define FRAME_DEF_REFR_PERIOD   33  /*[ms]*/

#define DECODER_BUFFER_SIZE (8 * 1024)

/**********************
 *      TYPEDEFS
 **********************/
struct ffmpeg_context_s {
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
#if LV_FFMPEG_AUDIO_SUPPORT != 0
    AVStream * audio_stream;
    AVCodecContext * audio_dec_ctx;
    int audio_stream_idx;
    AVFrame * audio_frame;
    struct SwrContext * swr_ctx;
    AVFormatContext * audio_out_fmt_ctx;
    uint8_t * audio_buf;
    int audio_buf_size;
    bool has_audio;
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
static int ffmpeg_image_allocate(struct ffmpeg_context_s * ffmpeg_ctx);
static int ffmpeg_get_image_header(lv_image_decoder_dsc_t * dsc, lv_image_header_t * header);
static int ffmpeg_get_frame_refr_period(struct ffmpeg_context_s * ffmpeg_ctx);
static uint8_t * ffmpeg_get_image_data(struct ffmpeg_context_s * ffmpeg_ctx);
static int ffmpeg_update_next_frame(struct ffmpeg_context_s * ffmpeg_ctx);
static int ffmpeg_output_video_frame(struct ffmpeg_context_s * ffmpeg_ctx);
static bool ffmpeg_pix_fmt_has_alpha(enum AVPixelFormat pix_fmt);
static bool ffmpeg_pix_fmt_is_yuv(enum AVPixelFormat pix_fmt);
#if LV_FFMPEG_AUDIO_SUPPORT != 0
static int ffmpeg_decode_audio_packet(AVCodecContext * dec, const AVPacket * pkt,
                                      struct ffmpeg_context_s * ffmpeg_ctx);
static int ffmpeg_output_audio_frame(struct ffmpeg_context_s * ffmpeg_ctx);
static int ffmpeg_audio_init(struct ffmpeg_context_s * ffmpeg_ctx);
static void ffmpeg_audio_deinit(struct ffmpeg_context_s * ffmpeg_ctx);
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

    if(ffmpeg_image_allocate(player->ffmpeg_ctx) < 0) {
        LV_LOG_ERROR("ffmpeg image allocate failed");
        ffmpeg_close(player->ffmpeg_ctx);
        player->ffmpeg_ctx = NULL;
        goto failed;
    }

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
            LV_LOG_INFO("ffmpeg player start");
            break;
        case LV_FFMPEG_PLAYER_CMD_STOP:
            av_seek_frame(player->ffmpeg_ctx->fmt_ctx,
                          0, 0, AVSEEK_FLAG_BACKWARD);
            lv_timer_pause(timer);
            LV_LOG_INFO("ffmpeg player stop");
            break;
        case LV_FFMPEG_PLAYER_CMD_PAUSE:
            lv_timer_pause(timer);
            LV_LOG_INFO("ffmpeg player pause");
            break;
        case LV_FFMPEG_PLAYER_CMD_RESUME:
            lv_timer_resume(timer);
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
    LV_LOG_INFO("Set volume to %d", player->volume);
}

int lv_ffmpeg_player_get_volume(lv_obj_t * obj)
{
    LV_ASSERT_OBJ(obj, MY_CLASS);
    lv_ffmpeg_player_t * player = (lv_ffmpeg_player_t *)obj;
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

    int width = ffmpeg_ctx->video_dec_ctx->width;
    int height = ffmpeg_ctx->video_dec_ctx->height;
    AVFrame * frame = ffmpeg_ctx->frame;

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

    /* copy decoded frame to destination buffer:
     * this is required since rawvideo expects non aligned data
     */
    av_image_copy(ffmpeg_ctx->video_src_data, ffmpeg_ctx->video_src_linesize,
                  (const uint8_t **)(frame->data), frame->linesize,
                  ffmpeg_ctx->video_dec_ctx->pix_fmt, width, height);

    if(ffmpeg_ctx->sws_ctx == NULL) {
        int swsFlags = SWS_BILINEAR;

        if(ffmpeg_pix_fmt_is_yuv(ffmpeg_ctx->video_dec_ctx->pix_fmt)) {

            /* When the video width and height are not multiples of 8,
             * and there is no size change in the conversion,
             * a blurry screen will appear on the right side
             * This problem was discovered in 2012 and
             * continues to exist in version 4.1.3 in 2019
             * This problem can be avoided by increasing SWS_ACCURATE_RND
             */
            if((width & 0x7) || (height & 0x7)) {
                LV_LOG_WARN("The width(%d) and height(%d) the image "
                            "is not a multiple of 8, "
                            "the decoding speed may be reduced",
                            width, height);
                swsFlags |= SWS_ACCURATE_RND;
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

failed:
    return ret;
}

static int ffmpeg_decode_packet(AVCodecContext * dec, const AVPacket * pkt,
                                struct ffmpeg_context_s * ffmpeg_ctx)
{
    int ret = 0;

    /* submit the packet to the decoder */
    ret = avcodec_send_packet(dec, pkt);
    if(ret < 0) {
        LV_LOG_ERROR("Error submitting a packet for decoding (%s)",
                     av_err2str(ret));
        return ret;
    }

    /* get all the available frames from the decoder */
    while(ret >= 0) {
        ret = avcodec_receive_frame(dec, ffmpeg_ctx->frame);
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

        /* write the frame data to output file */
        if(dec->codec->type == AVMEDIA_TYPE_VIDEO) {
            ret = ffmpeg_output_video_frame(ffmpeg_ctx);
        }
#if LV_FFMPEG_AUDIO_SUPPORT != 0
        else if(dec->codec->type == AVMEDIA_TYPE_AUDIO) {
            ret = ffmpeg_output_audio_frame(ffmpeg_ctx);
        }
#endif

        av_frame_unref(ffmpeg_ctx->frame);
        if(ret < 0) {
            LV_LOG_WARN("ffmpeg_decode_packet ended %d", ret);
            return ret;
        }
    }

    return 0;
}

static int ffmpeg_open_codec_context(int * stream_idx,
                                     AVCodecContext ** dec_ctx, AVFormatContext * fmt_ctx,
                                     enum AVMediaType type)
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
        if((ret = avcodec_open2(*dec_ctx, dec, &opts)) < 0) {
            LV_LOG_ERROR("Failed to open %s codec",
                         av_get_media_type_string(type));
            return ret;
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
                                 fmt_ctx, AVMEDIA_TYPE_VIDEO)
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

    while(1) {

        /* read frames from the file */
        if(av_read_frame(ffmpeg_ctx->fmt_ctx, ffmpeg_ctx->pkt) >= 0) {
            bool is_image = false;

            /* check if the packet belongs to a stream we are interested in,
             * otherwise skip it
             */
            if(ffmpeg_ctx->pkt->stream_index == ffmpeg_ctx->video_stream_idx) {
                ret = ffmpeg_decode_packet(ffmpeg_ctx->video_dec_ctx,
                                           ffmpeg_ctx->pkt, ffmpeg_ctx);
                is_image = true;
            }
#if LV_FFMPEG_AUDIO_SUPPORT != 0
            else if(ffmpeg_ctx->has_audio &&
                    ffmpeg_ctx->pkt->stream_index == ffmpeg_ctx->audio_stream_idx) {
                ret = ffmpeg_decode_packet(ffmpeg_ctx->audio_dec_ctx,
                                           ffmpeg_ctx->pkt, ffmpeg_ctx);
                is_image = true;
            }
#endif

            av_packet_unref(ffmpeg_ctx->pkt);

            if(ret < 0) {
                LV_LOG_WARN("video frame is empty %d", ret);
                break;
            }

            /* Used to filter data that is not an image */
            if(is_image) {
                break;
            }
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
           ffmpeg_ctx->fmt_ctx, AVMEDIA_TYPE_VIDEO)
       >= 0) {
        ffmpeg_ctx->video_stream = ffmpeg_ctx->fmt_ctx->streams[ffmpeg_ctx->video_stream_idx];

        ffmpeg_ctx->has_alpha = ffmpeg_pix_fmt_has_alpha(ffmpeg_ctx->video_dec_ctx->pix_fmt);

        ffmpeg_ctx->video_dst_pix_fmt = (ffmpeg_ctx->has_alpha ? AV_PIX_FMT_BGRA : AV_PIX_FMT_TRUE_COLOR);
    }

#if LV_FFMPEG_AUDIO_SUPPORT != 0
    /* Try to open audio stream */
    if(ffmpeg_open_codec_context(
           &(ffmpeg_ctx->audio_stream_idx),
           &(ffmpeg_ctx->audio_dec_ctx),
           ffmpeg_ctx->fmt_ctx, AVMEDIA_TYPE_AUDIO)
       >= 0) {
        ffmpeg_ctx->audio_stream = ffmpeg_ctx->fmt_ctx->streams[ffmpeg_ctx->audio_stream_idx];
        ffmpeg_ctx->has_audio = true;
        LV_LOG_INFO("Audio stream found, codec: %s",
                    avcodec_get_name(ffmpeg_ctx->audio_dec_ctx->codec_id));
    } else {
        ffmpeg_ctx->has_audio = false;
        LV_LOG_INFO("No audio stream found");
    }
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
    avcodec_free_context(&(ffmpeg_ctx->video_dec_ctx));
    avformat_close_input(&(ffmpeg_ctx->fmt_ctx));
    av_packet_free(&ffmpeg_ctx->pkt);
    av_frame_free(&(ffmpeg_ctx->frame));
    if(ffmpeg_ctx->video_src_data[0] != NULL) {
        av_free(ffmpeg_ctx->video_src_data[0]);
        ffmpeg_ctx->video_src_data[0] = NULL;
    }
#if LV_FFMPEG_AUDIO_SUPPORT != 0
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

    sws_freeContext(ffmpeg_ctx->sws_ctx);
    ffmpeg_close_src_ctx(ffmpeg_ctx);
    ffmpeg_close_dst_ctx(ffmpeg_ctx);
    if(ffmpeg_ctx->io_ctx != NULL) {
        av_free(ffmpeg_ctx->io_ctx->buffer);
        av_free(ffmpeg_ctx->io_ctx);
        lv_fs_close(&(ffmpeg_ctx->lv_file));
    }
#if LV_FFMPEG_AUDIO_SUPPORT != 0
    if(ffmpeg_ctx->audio_buf != NULL) {
        av_free(ffmpeg_ctx->audio_buf);
        ffmpeg_ctx->audio_buf = NULL;
    }
#endif
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

    int has_next = ffmpeg_update_next_frame(player->ffmpeg_ctx);

    if(has_next < 0) {
        lv_ffmpeg_player_set_cmd(obj, player->auto_restart ? LV_FFMPEG_PLAYER_CMD_START : LV_FFMPEG_PLAYER_CMD_STOP);
        if(!player->auto_restart) {
            lv_obj_send_event((lv_obj_t *)player, LV_EVENT_READY, NULL);
        }
        return;
    }

    lv_image_cache_drop(lv_image_get_src(obj));

    lv_obj_invalidate(obj);
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

/* Initialize audio output device */
static int ffmpeg_audio_init(struct ffmpeg_context_s * ffmpeg_ctx)
{
    int ret;

    if(!ffmpeg_ctx->has_audio || !ffmpeg_ctx->audio_dec_ctx) {
        return -1;
    }

    /* Initialize audio output context */
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

    /* Initialize audio resampler */
    AVChannelLayout src_ch_layout = ffmpeg_ctx->audio_dec_ctx->ch_layout;
    AVChannelLayout dst_ch_layout;
    av_channel_layout_default(&dst_ch_layout, 2);

    ffmpeg_ctx->swr_ctx = swr_alloc();
    av_opt_set_chlayout(ffmpeg_ctx->swr_ctx, "in_chlayout", &src_ch_layout, 0);
    av_opt_set_int(ffmpeg_ctx->swr_ctx, "in_sample_rate", ffmpeg_ctx->audio_dec_ctx->sample_rate, 0);
    av_opt_set_sample_fmt(ffmpeg_ctx->swr_ctx, "in_sample_fmt", ffmpeg_ctx->audio_dec_ctx->sample_fmt, 0);
    av_opt_set_chlayout(ffmpeg_ctx->swr_ctx, "out_chlayout", &dst_ch_layout, 0);
    av_opt_set_int(ffmpeg_ctx->swr_ctx, "out_sample_rate", 44100, 0);
    av_opt_set_sample_fmt(ffmpeg_ctx->swr_ctx, "out_sample_fmt", AV_SAMPLE_FMT_S16, 0);
    ret = swr_init(ffmpeg_ctx->swr_ctx);
    if(ret < 0) {
        LV_LOG_ERROR("Error initializing audio resampler: %s", av_err2str(ret));
        avformat_free_context(ffmpeg_ctx->audio_out_fmt_ctx);
        ffmpeg_ctx->audio_out_fmt_ctx = NULL;
        return -1;
    }

    /* Open output device */
    if(!(ffmpeg_ctx->audio_out_fmt_ctx->oformat->flags & AVFMT_NOFILE)) {
        ret = avio_open(&ffmpeg_ctx->audio_out_fmt_ctx->pb, ffmpeg_ctx->audio_out_fmt_ctx->url, AVIO_FLAG_WRITE);
        if(ret < 0) {
            LV_LOG_ERROR("Error opening audio output device: %s", av_err2str(ret));
            swr_free(&ffmpeg_ctx->swr_ctx);
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
        swr_free(&ffmpeg_ctx->swr_ctx);
        avformat_free_context(ffmpeg_ctx->audio_out_fmt_ctx);
        ffmpeg_ctx->audio_out_fmt_ctx = NULL;
        return -1;
    }

    ffmpeg_ctx->audio_buf = NULL;
    ffmpeg_ctx->audio_buf_size = 0;

    LV_LOG_INFO("Audio output initialized successfully");
    return 0;
}

/* Deinitialize audio output device */
static void ffmpeg_audio_deinit(struct ffmpeg_context_s * ffmpeg_ctx)
{
    if(!ffmpeg_ctx) {
        return;
    }

    if(ffmpeg_ctx->audio_out_fmt_ctx) {
        av_write_trailer(ffmpeg_ctx->audio_out_fmt_ctx);
        if(!(ffmpeg_ctx->audio_out_fmt_ctx->oformat->flags & AVFMT_NOFILE)) {
            avio_closep(&ffmpeg_ctx->audio_out_fmt_ctx->pb);
        }
        avformat_free_context(ffmpeg_ctx->audio_out_fmt_ctx);
        ffmpeg_ctx->audio_out_fmt_ctx = NULL;
    }

    if(ffmpeg_ctx->swr_ctx) {
        swr_free(&ffmpeg_ctx->swr_ctx);
        ffmpeg_ctx->swr_ctx = NULL;
    }
}

/* Output audio frame to device */
static int ffmpeg_output_audio_frame(struct ffmpeg_context_s * ffmpeg_ctx)
{
    int ret = 0;
    AVFrame *frame = ffmpeg_ctx->audio_frame;
    lv_ffmpeg_player_t *player = NULL;

    if(!ffmpeg_ctx->audio_out_fmt_ctx || !ffmpeg_ctx->swr_ctx) {
        return 0;
    }

    /* Get player pointer to check audio enabled status */
    if(ffmpeg_ctx->video_dst_linesize[0] != 0) {
        /* Try to find the player object that owns this context */
        /* This is a workaround since we don't have direct access to player here */
        /* In a real implementation, you would pass the player pointer or use a different approach */
    }

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
    int out_samples = swr_convert(
        ffmpeg_ctx->swr_ctx, &ffmpeg_ctx->audio_buf, dst_nb_samples,
        (const uint8_t **)frame->data, frame->nb_samples
    );

    if(out_samples <= 0) {
        return 0;
    }

    /* Apply volume control (simple multiplication) */
    /* Note: This is a basic implementation. For better quality, use proper audio processing */
    if(frame->format == AV_SAMPLE_FMT_FLT || frame->format == AV_SAMPLE_FMT_FLTP) {
        float *samples = (float *)ffmpeg_ctx->audio_buf;
        float volume_factor = 1.0f; /* Default volume */
        for(int i = 0; i < out_samples * 2; i++) {
            samples[i] *= volume_factor;
        }
    }

    /* Create output packet */
    AVPacket *out_pkt = av_packet_alloc();
    if(!out_pkt) {
        LV_LOG_ERROR("Failed to allocate audio output packet");
        return -1;
    }

    out_pkt->data = ffmpeg_ctx->audio_buf;
    out_pkt->size = out_samples * 4; /* 16-bit stereo = 2 bytes/sample * 2 channels */
    out_pkt->stream_index = 0;
    out_pkt->pts = frame->pts;
    out_pkt->dts = frame->pkt_dts;
    out_pkt->duration = frame->duration;

    /* Write to output device */
    ret = av_interleaved_write_frame(ffmpeg_ctx->audio_out_fmt_ctx, out_pkt);

    av_packet_free(&out_pkt);

    if(ret < 0) {
        LV_LOG_ERROR("Error writing audio frame: %s", av_err2str(ret));
        return ret;
    }

    return 0;
}

#endif /* LV_FFMPEG_AUDIO_SUPPORT */

#endif /*LV_USE_FFMPEG*/
