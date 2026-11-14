/*
 * audio.h — minimal audio playback of demodulated mono audio through
 * the sound card (CoreAudio AudioQueue on macOS).
 */
#ifndef TE15_AUDIO_H
#define TE15_AUDIO_H

#include <stdbool.h>
#include <stddef.h>

typedef struct AudioPlayer AudioPlayer;

/*
 * Create a player from mono float audio (values outside [-1,1] are
 * clamped) at `sample_rate` Hz. Does not touch the audio device until
 * audio_player_play() is called. Returns NULL on allocation failure.
 */
AudioPlayer *audio_player_create(const float *mono, size_t n, int sample_rate);

/* Stop (synchronously) and release the player. */
void audio_player_destroy(AudioPlayer *p);

/*
 * Start or resume playback. Returns false when no audio output device
 * is available. Starting from the end restarts at the beginning.
 */
bool audio_player_play(AudioPlayer *p);

/* Pause; the position is retained. */
void audio_player_pause(AudioPlayer *p);

/* Seek to `seconds` (clamped to [0, duration]); restarts playback if it
 * was playing. */
void audio_player_seek(AudioPlayer *p, double seconds);

/* Current position in seconds (advances during playback). */
double audio_player_position(AudioPlayer *p);

/* Total duration in seconds. */
double audio_player_duration(AudioPlayer *p);

/* True while playing. */
bool audio_player_is_playing(AudioPlayer *p);

/* True once playback has reached the end of the buffer. */
bool audio_player_finished(AudioPlayer *p);

#endif /* TE15_AUDIO_H */
