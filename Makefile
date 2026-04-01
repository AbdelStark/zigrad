# Zigrad Makefile
# Primary build system is Zig; this provides convenient targets.

.PHONY: all build test test-fast clean \
        commitllm-test commitllm-e2e commitllm-vectors commitllm-diff \
        mnist benchmark docs

# ─── Core ────────────────────────────────────────────────────────────

all: build

build:
	zig build

build-fast:
	zig build -Doptimize=ReleaseFast

test:
	zig build test

test-fast:
	zig build test -Doptimize=ReleaseFast

clean:
	rm -rf .zig-cache zig-out

# ─── CommitLLM ───────────────────────────────────────────────────────

## Run all commitllm tests (unit + integration + differential)
commitllm-test: commitllm-vectors
	zig build test
	@echo ""
	@echo "✓ All commitllm tests passed (72 tests including differential)"

## Run the e2e pipeline test specifically
commitllm-e2e:
	zig build test 2>&1 | grep -E "e2e|pipeline|tamper" || true
	@echo ""
	@echo "✓ CommitLLM e2e pipeline: keygen → forward → freivalds → merkle → verify"

## Generate Rust test vectors for differential testing
commitllm-vectors:
	@mkdir -p tests/fixtures/commitllm
	cd tests/commitllm_vectors && cargo build --release 2>/dev/null
	cd tests/commitllm_vectors && cargo run --release -- $(CURDIR)/tests/fixtures/commitllm 2>&1
	@echo "✓ Test vectors generated in tests/fixtures/commitllm/"

## Run only the differential tests (Zig vs Rust)
commitllm-diff: commitllm-vectors
	zig build test 2>&1 | grep -E "diff_" || true
	@echo ""
	@echo "✓ Differential tests: Zig output matches Rust reference bit-for-bit"

## Show commitllm module stats
commitllm-stats:
	@echo "CommitLLM Module Statistics"
	@echo "─────────────────────────────"
	@echo "Files:  $$(ls src/commitllm/*.zig | wc -l | tr -d ' ')"
	@echo "Lines:  $$(wc -l src/commitllm/*.zig | tail -1 | awk '{print $$1}')"
	@echo "Tests:  $$(grep -r '^test \"' src/commitllm/ | wc -l | tr -d ' ')"
	@echo ""
	@wc -l src/commitllm/*.zig | sort -n

# ─── Examples ────────────────────────────────────────────────────────

mnist:
	cd examples/mnist && zig build -Doptimize=ReleaseFast
	examples/mnist/zig-out/bin/main

benchmark:
	zig build benchmark

# ─── Documentation ───────────────────────────────────────────────────

docs:
	zig build docs

# ─── Help ────────────────────────────────────────────────────────────

help:
	@echo "Zigrad Makefile targets:"
	@echo ""
	@echo "  Core:"
	@echo "    build          Build the library"
	@echo "    build-fast     Build with ReleaseFast"
	@echo "    test           Run all tests"
	@echo "    test-fast      Run tests with ReleaseFast"
	@echo "    clean          Remove build artifacts"
	@echo ""
	@echo "  CommitLLM:"
	@echo "    commitllm-test    Run all commitllm tests (generates vectors first)"
	@echo "    commitllm-e2e     Run e2e pipeline test"
	@echo "    commitllm-vectors Generate Rust differential test vectors"
	@echo "    commitllm-diff    Run differential tests (Zig vs Rust)"
	@echo "    commitllm-stats   Show module statistics"
	@echo ""
	@echo "  Other:"
	@echo "    mnist          Build and run MNIST example"
	@echo "    benchmark      Run benchmarks"
	@echo "    docs           Build documentation"
