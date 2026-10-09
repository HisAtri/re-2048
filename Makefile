# re-2048 的唯一入口。
#
# 为什么要有这一层：C++ 是 CMake、Python 是 uv，而虚拟环境、构建产物、跑谱数据
# 又都在仓库之外的 Linux 文件系统上（仓库位于 /mnt/e 的 9p 挂载，编译 IO 与
# 大规模写盘都明显更慢）。散着敲命令迟早会出现“改了文件忘了重建”或者
# “用错解释器”的问题，因此日常动作统一从这里走。

SHELL := /bin/bash
ROOT  := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))

# 构建与虚拟环境默认放在仓库外；数据和模型默认归档在仓库的 output/。
WORK   ?= $(HOME)/r2048
BUILD  ?= $(WORK)/build
OUTPUT ?= $(ROOT)/output
VENV   ?= $(WORK)/venv

PY    := $(VENV)/bin/python
JOBS  ?= $(shell nproc)
PYEXT := $(BUILD)/pyext
BIN   := $(BUILD)/bin/r2048

export UV_PROJECT_ENVIRONMENT := $(VENV)
export R2048_OUTPUT_DIR       := $(OUTPUT)
export R2048_TEST_BINARY      := $(BUILD)/bin/r2048_tests
# CLI 单独导出：有些测试要跑真实子命令（例如“C++ 写出的逐局记录必须满足 Python 侧的
# 协议清单”），不能靠从 R2048_TEST_BINARY 反推路径。
export R2048_CLI              := $(BIN)

.DEFAULT_GOAL := help

.PHONY: help env-check env configure build test test-cpp test-py lint fmt run py bench webui clean

help:
	@echo "re-2048 构建入口"
	@echo ""
	@echo "  make env        创建/更新 Python 环境（uv sync，会下载 torch，只需跑一次）"
	@echo "  make env-check  只检查工具链与路径，不改动任何东西"
	@echo "  make build      配置并构建 C++（库 + CLI + 测试 + Python 扩展）"
	@echo "  make test       跑 C++ 与 Python 全部测试"
	@echo "  make run ARGS=\"generate --policy random ...\"  运行 C++ CLI"
	@echo "  make py  ARGS=\"...\"                          运行 Python 侧（python -m r2048）"
	@echo "  make webui      启动 Gradio 交互界面（可选组件，见 ADR 0039）"
	@echo "  make fmt        格式化 C++（clang-format）与 Python（ruff）"
	@echo "  make clean      删除构建目录（不动数据）"
	@echo ""
	@echo "路径：BUILD=$(BUILD)  OUTPUT=$(OUTPUT)  VENV=$(VENV)"

env-check:
	@echo "仓库根目录 : $(ROOT)"
	@echo "构建目录   : $(BUILD)"
	@echo "输出目录   : $(OUTPUT)"
	@echo "虚拟环境   : $(VENV)"
	@echo ""
	@missing=0; \
	for tool in uv g++ cmake ninja; do \
	  if command -v $$tool >/dev/null 2>&1; then \
	    printf "  [ok]   %-8s %s\n" "$$tool" "$$(command -v $$tool)"; \
	  else \
	    printf "  [缺失] %-8s\n" "$$tool"; missing=1; \
	  fi; \
	done; \
	if command -v ccache >/dev/null 2>&1; then printf "  [ok]   %-8s %s\n" ccache "$$(command -v ccache)"; \
	else printf "  [可选] %-8s 未安装（只影响重复编译速度）\n" ccache; fi; \
	if [ $$missing -ne 0 ]; then \
	  echo ""; \
	  echo "补齐工具链："; \
	  echo "  uv tool install cmake      # 比 Ubuntu 22.04 自带的 3.22 新"; \
	  echo "  uv tool install ninja"; \
	  echo "  sudo apt-get install -y ccache   # 可选"; \
	  exit 1; \
	fi
	@echo ""
	@if [ -d "$(VENV)" ]; then echo "虚拟环境已就绪（$(VENV)）"; else echo "虚拟环境尚未创建：先跑 make env"; fi

env: env-check
	cd $(ROOT) && uv sync
	@if [ ! -e "$(ROOT)/.venv" ]; then ln -s "$(VENV)" "$(ROOT)/.venv"; echo "已创建 .venv 符号链接 -> $(VENV)"; fi
	@echo "环境就绪。下一步：make build"

configure: env-check
	@test -x "$(PY)" || { echo "虚拟环境不完整（缺 $(PY)）：先跑 make env"; exit 1; }
	cmake -S $(ROOT) -B $(BUILD) -G Ninja \
	  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
	  -DR2048_NATIVE_ARCH=ON \
	  -DR2048_BUILD_PYEXT=ON \
	  -DPython3_EXECUTABLE=$(PY) \
	  -Dpybind11_DIR="$$($(PY) -m pybind11 --cmakedir)"

build: configure
	cmake --build $(BUILD) -j$(JOBS)

test: build test-cpp test-py

test-cpp:
	ctest --test-dir $(BUILD) --output-on-failure

test-py: build
	cd $(ROOT) && PYTHONPATH=$(PYEXT) uv run pytest

lint:
	cd $(ROOT) && uv run ruff check python

fmt:
	@if command -v clang-format >/dev/null 2>&1; then \
	  find $(ROOT)/cpp -name '*.cpp' -o -name '*.h' -o -name '*.h.in' | xargs clang-format -i; \
	  echo "已格式化 C++ 源码"; \
	else \
	  echo "跳过 C++ 格式化：未安装 clang-format（sudo apt-get install -y clang-format）"; \
	fi
	cd $(ROOT) && uv run ruff format python && uv run ruff check --fix python

run: build
	$(BIN) $(ARGS)

py:
	cd $(ROOT) && PYTHONPATH=$(PYEXT) uv run python -m r2048 $(ARGS)

bench: build
	$(BIN) bench $(ARGS)

# WebUI 是可选组件（ADR 0039）：gradio 在 `webui` 依赖组里，因而不进核心依赖。
# `uv run --group webui` 会按需把它装进环境。默认只绑 127.0.0.1；换端口用 ARGS="--port 8000"。
webui: build
	cd $(ROOT) && PYTHONPATH=$(PYEXT) uv run --group webui python -m r2048 webui $(ARGS)

clean:
	rm -rf $(BUILD)
	@echo "已删除构建目录 $(BUILD)（数据与虚拟环境未受影响）"
