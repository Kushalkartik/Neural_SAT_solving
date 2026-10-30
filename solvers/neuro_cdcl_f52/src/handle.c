#include <unistd.h>
#include "handle.h"

#include <assert.h>
#include <stdbool.h>
#include <stdio.h>

static void (*handler) (int);
static volatile int caught_signal;
static volatile bool handler_set;

// clang-format off

#define SIGNAL(SIG) \
static void (*SIG ## _handler)(int);
SIGNALS
#undef SIGNAL

void
kissat_reset_signal_handler (void)
{
  if (!handler_set)
    return;
#define SIGNAL(SIG) \
  signal (SIG, SIG ## _handler);
  SIGNALS
#undef SIGNAL
  handler_set = false;
  handler = 0;
}

// clang-format on

static void catch_signal (int sig) {
  if (caught_signal)
    return;
  caught_signal = sig;
  assert (handler_set);
  assert (handler);
  handler (sig);
  kissat_reset_signal_handler ();
  raise (sig);
}

void kissat_init_signal_handler (void (*h) (int sig)) {
  assert (!handler);
  handler = h;
  handler_set = true;
#define SIGNAL(SIG) SIG##_handler = signal (SIG, catch_signal);
  SIGNALS
#undef SIGNAL
}

#ifdef _WIN32
#include <windows.h>
static HANDLE kissat_win_timer_handle = NULL;
static void (*volatile kissat_win_alarm_handler) (void) = NULL;

static VOID CALLBACK KissatWinAlarmCallback (PVOID lpParameter, BOOLEAN TimerOrWaitFired) {
  (void) lpParameter; (void) TimerOrWaitFired;
  if (kissat_win_alarm_handler)
    kissat_win_alarm_handler ();
}

void kissat_init_alarm (void (*handler) (void)) {
  kissat_win_alarm_handler = handler;
}

void kissat_start_alarm (int seconds) {
  if (seconds <= 0)
    return;
  if (kissat_win_timer_handle) {
    DeleteTimerQueueTimer (NULL, kissat_win_timer_handle, NULL);
    kissat_win_timer_handle = NULL;
  }
  CreateTimerQueueTimer (&kissat_win_timer_handle, NULL, KissatWinAlarmCallback,
                         NULL, (DWORD)(seconds * 1000), 0, WT_EXECUTEONLYONCE);
}

void kissat_reset_alarm (void) {
  if (kissat_win_timer_handle) {
    DeleteTimerQueueTimer (NULL, kissat_win_timer_handle, NULL);
    kissat_win_timer_handle = NULL;
  }
}
#else
static volatile bool caught_alarm;
static volatile bool alarm_handler_set;
static void (*volatile SIGALRM_handler) (int);
static void (*volatile handle_alarm) (void);

static void catch_alarm (int sig) {
  assert (sig == SIGALRM);
  if (caught_alarm)
    return;
  caught_alarm = true;
  static void (*volatile handler) (void);
  handler = handle_alarm;
  if (!alarm_handler_set)
    raise (sig);
  assert (handler);
  handler ();
}

void kissat_init_alarm (void (*handler) (void)) {
  assert (handler);
  assert (!caught_alarm);
  handle_alarm = handler;
  alarm_handler_set = true;
  assert (!SIGALRM_handler);
  SIGALRM_handler = signal (SIGALRM, catch_alarm);
}

void kissat_start_alarm (int seconds) {
  alarm (seconds);
}

void kissat_reset_alarm (void) {
  assert (alarm_handler_set);
  assert (handle_alarm);
  alarm_handler_set = false;
  handle_alarm = 0;
  (void) signal (SIGALRM, SIGALRM_handler);
}
#endif
