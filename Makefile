# Makefile for ToxicElement15 — SDR waterfall analyzer (GTK3 + FFTW3f).
#
# Targets:
#   make              build ./build/te15
#   make test         build and run the unit tests (no GUI)
#   make check FILE=  build and run the headless end-to-end check
#   make gen-test-data  generate data/gqrx_..._2000000_fc.raw (48 MB)
#   make run FILE=    build and launch the GUI on a recording
#   make clean        remove build artifacts

CC       ?= cc
CFLAGS   ?= -O3
WARNINGS := -Wall -Wextra -Wshadow
CPPFLAGS := -Isrc
GTK_CFLAGS := $(shell pkg-config --cflags gtk+-3.0)
GTK_LIBS   := $(shell pkg-config --libs gtk+-3.0)
FFTW_CFLAGS := $(shell pkg-config --cflags fftw3f)
FFTW_LIBS   := $(shell pkg-config --libs fftw3f)
JSONC_CFLAGS := $(shell pkg-config --cflags json-c)
JSONC_LIBS   := $(shell pkg-config --libs json-c)
AUDIO_LIBS := -framework AudioToolbox -framework CoreAudio \
              -framework CoreFoundation
LDLIBS   := $(GTK_LIBS) $(FFTW_LIBS) $(JSONC_LIBS) $(AUDIO_LIBS) -lpthread -lm
ALL_CFLAGS := $(CFLAGS) $(WARNINGS) $(CPPFLAGS) $(GTK_CFLAGS) $(FFTW_CFLAGS) $(JSONC_CFLAGS)

APP_SRC := \
	src/main.c \
	src/iq_source.c \
	src/dsp.c \
	src/colormap.c \
	src/signals.c \
	src/fit.c \
	src/region.c \
	src/detect.c \
	src/waterfall_cache.c \
	src/streambuf.c \
	src/renderbuf.c \
	src/fcdataset.c \
	src/fcdrift.c \
	src/fclattice.c \
	src/fclk.c \
	src/dagcore.c \
	src/dag.c \
	src/fcdag.c \
	src/fcreport.c \
	src/lte_uplink.c \
	src/analysis.c \
	src/metrics.c \
	src/wav.c \
	src/audio.c \
	src/audio_file.c \
	src/cyclic.c \
	src/fluct.c \
	src/cpar.c \
	src/hos.c \
	src/ssca.c \
	src/cycledet.c \
	src/ccc.c \
	src/sptk.c \
	src/sdist.c \
	src/classic.c \
	src/filters.c \
	src/pulse.c \
	src/modconf.c \
	src/intraburst.c \
	src/ssclass.c \
	src/sot.c \
	src/apublish.c \
	src/director.c \
	src/report.c \
	src/aconfig.c \
	src/aregistry.c \
	src/aresult.c \
	src/synth.c \
	src/ui.c

TEST_SRC := \
	tests/test_runner.c \
	src/iq_source.c \
	src/dsp.c \
	src/colormap.c \
	src/signals.c \
	src/fit.c \
	src/region.c \
	src/detect.c \
	src/waterfall_cache.c \
	src/streambuf.c \
	src/renderbuf.c \
	src/fcdataset.c \
	src/fcdrift.c \
	src/fclattice.c \
	src/fclk.c \
	src/dagcore.c \
	src/fcdag.c \
	src/fcreport.c \
	src/lte_uplink.c \
	src/analysis.c \
	src/metrics.c \
	src/wav.c \
	src/audio.c \
	src/audio_file.c \
	src/cyclic.c \
	src/fluct.c \
	src/cpar.c \
	src/hos.c \
	src/ssca.c \
	src/cycledet.c \
	src/ccc.c \
	src/sptk.c \
	src/sdist.c \
	src/classic.c \
	src/filters.c \
	src/pulse.c \
	src/modconf.c \
	src/intraburst.c \
	src/ssclass.c \
	src/sot.c \
	src/apublish.c \
	src/director.c \
	src/report.c \
	src/aconfig.c \
	src/aregistry.c \
	src/dag.c \
	src/aresult.c \
	src/synth.c

TEST_DATA := data/gqrx_20260306_174402_2530000000_2000000_fc.raw

.PHONY: all test golden check run gen-test-data clean

all: build/te15

ABOUT_IMAGE := build/about_image.c

# The About artwork is embedded in the executable (generated from the JPEG
# at build time; nothing is read from media/ at run time).
$(ABOUT_IMAGE): media/about.gif tools/bin2c.c | build
	$(CC) -O1 -o build/bin2c tools/bin2c.c
	./build/bin2c media/about.gif te15_about_image > $@.tmp && mv $@.tmp $@

build/te15: $(APP_SRC) $(ABOUT_IMAGE) src/*.h | build
	$(CC) $(ALL_CFLAGS) -o $@ $(APP_SRC) $(ABOUT_IMAGE) $(LDLIBS)

build/test_runner: $(TEST_SRC) src/*.h | build
	$(CC) $(ALL_CFLAGS) -o $@ $(TEST_SRC) $(FFTW_LIBS) $(JSONC_LIBS) $(AUDIO_LIBS) -lpthread -lm

build:
	mkdir -p build

golden: build/golden_report
	./build/golden_report tests/golden/report_synth.txt

GOLDEN_SRC := $(filter-out src/main.c src/ui.c,$(APP_SRC))

build/golden_report: tests/golden_report.c $(GOLDEN_SRC) $(ABOUT_IMAGE) src/*.h | build
	$(CC) $(ALL_CFLAGS) -o $@ tests/golden_report.c $(GOLDEN_SRC) $(ABOUT_IMAGE) $(LDLIBS)

test: build/test_runner
	./build/test_runner

check: build/te15
	@test -n "$(FILE)" || (echo "usage: make check FILE=<recording>"; exit 2)
	./build/te15 --check "$(FILE)"

run: build/te15
	@test -n "$(FILE)" || (echo "usage: make run FILE=<recording>"; exit 2)
	./build/te15 "$(FILE)"

gen-test-data:
	python3 tools/gen_test_iq.py "$(TEST_DATA)"

clean:
	rm -rf build
