# Makefile — Terncore Metal kernel build
# Builds libternary.dylib from Objective-C + Metal source
#
# Usage:
#   make          — build shared library
#   make test     — run correctness test
#   make bench    — run benchmark
#   make clean    — remove build artifacts

CC       = clang
CFLAGS   = -O2 -Wall -Wno-deprecated-declarations -fobjc-arc -fPIC
LDFLAGS  = -dynamiclib -framework Metal -framework Foundation

# Metal compiler — prefer Xcode toolchain, fall back to xcrun
XCODE_TC = /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin
METAL    = $(if $(wildcard $(XCODE_TC)/metal),$(XCODE_TC)/metal,xcrun -sdk macosx metal)
METALLIB = $(if $(wildcard $(XCODE_TC)/metallib),$(XCODE_TC)/metallib,xcrun -sdk macosx metallib)

SRC_DIR  = .
BUILD    = build
LIB      = $(BUILD)/libternary.dylib
METALBIN = $(BUILD)/ternary_matmul.metallib

.PHONY: all clean test bench

all: $(LIB)

# Optional: pre-compile Metal shaders (requires Metal Toolchain)
metallib: $(METALBIN)

$(BUILD):
	mkdir -p $(BUILD)

# Compile Metal shaders to .air then .metallib
$(BUILD)/ternary_matmul.air: $(SRC_DIR)/ternary_matmul.metal | $(BUILD)
	$(METAL) -c -std=metal3.1 -O2 -o $@ $<

$(METALBIN): $(BUILD)/ternary_matmul.air
	$(METALLIB) -o $@ $<

# Compile Objective-C engine to shared library
$(LIB): $(SRC_DIR)/ternary_engine.m $(SRC_DIR)/ternary_engine.h | $(BUILD)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(SRC_DIR)/ternary_engine.m

test: $(LIB)
	cd $(CURDIR) && python3 ternary_metal.py --test

bench: $(LIB)
	cd $(CURDIR) && python3 benchmark_metal.py

clean:
	rm -rf $(BUILD)
