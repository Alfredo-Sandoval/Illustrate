# Template preset: base/python-first quality gates.
.PHONY: env setup bootstrap start host-check package loc lint typecheck test test-cov qa docs-build docs-serve docs-check clean

ENV_ARGS ?=
PYTHON ?= python
MKDOCS ?= mkdocs
DOCS_WARN_ENV ?= NO_MKDOCS_2_WARNING=1
SOURCE_DIRS ?= illustrate tests
EXCLUDE_PATHS ?= .git .venv venv env node_modules .next .turbo out __pycache__ *.egg-info .eggs .pytest_cache .ruff_cache .mypy_cache .cache build dist htmlcov coverage .coverage site results data external
LOC_PRUNE_NAMES := $(foreach p,$(EXCLUDE_PATHS),-name '$(p)' -o ) -false
LOC_GROUP_DEPTH ?= 2

env:
	@test -f environment/setup.sh || { echo "Missing setup script: environment/setup.sh"; exit 1; }
	bash environment/setup.sh $(ENV_ARGS)

setup: env

bootstrap: env

start:
	@test -f environment/setup.sh || { echo "Missing setup script: environment/setup.sh"; exit 1; }
	bash environment/setup.sh --launch $(ENV_ARGS)

host-check:
	@test -f environment/setup.sh || { echo "Missing setup script: environment/setup.sh"; exit 1; }
	bash environment/setup.sh --host-check

package:
	@test -f packaging/macos/build_app.sh || { echo "Missing script: packaging/macos/build_app.sh"; exit 1; }
	bash packaging/macos/build_app.sh

lint:
	ruff check .

typecheck:
	@command -v node >/dev/null 2>&1 || { echo "Missing dependency: Node.js on PATH is required for Pyright."; exit 1; }
	@python_path=$$($(PYTHON) -c 'import sys; print(sys.executable)') && \
		PYRIGHT_PYTHON_GLOBAL_NODE=on PYRIGHT_PYTHON_NODEJS_WHEEL=off $(PYTHON) -m pyright --pythonpath "$$python_path"

test:
	pytest

test-cov:
	pytest --cov=illustrate --cov-report=term-missing

qa: lint typecheck test

docs-build:
	@command -v $(MKDOCS) >/dev/null 2>&1 || { echo "Missing dependency: $(MKDOCS). Run 'make env' to install docs tooling."; exit 1; }
	$(DOCS_WARN_ENV) $(MKDOCS) build --strict

docs-serve:
	@command -v $(MKDOCS) >/dev/null 2>&1 || { echo "Missing dependency: $(MKDOCS). Run 'make env' to install docs tooling."; exit 1; }
	$(DOCS_WARN_ENV) $(MKDOCS) serve

docs-check: docs-build

# LOC summary: module breakdown (by depth) + language breakdown + file count.
loc:
	@paths=""; \
	for d in $(SOURCE_DIRS); do \
		if [ -d "$$d" ]; then \
			if [ -z "$$paths" ]; then paths="$$d"; else paths="$$paths $$d"; fi; \
		fi; \
	done; \
	if [ -z "$$paths" ]; then \
		echo "No source directories found in SOURCE_DIRS=$(SOURCE_DIRS)"; \
		exit 1; \
	fi; \
	{ \
		for d in $$paths; do \
			find "$$d" -type d \( $(LOC_PRUNE_NAMES) \) -prune -o -type f -name '*.py' -print0; \
		done; \
	} \
	| xargs -0 wc -l \
	| awk -v depth="$(LOC_GROUP_DEPTH)" ' \
		NF==2 && $$2!="total"{ \
			count=$$1; path=$$2; files++; \
			ne=split(path,ep,"."); ext=ep[ne]; lang[ext]+=count; \
			n=split(path,parts,"/"); if(n<2) next; \
			gd=depth; if(gd<1) gd=1; lim=gd; if(n-1<lim) lim=n-1; \
			key=parts[1]; for(i=2;i<=lim;i++) key=key"/"parts[i]; \
			mod[key]+=count; fc[key]++; total+=count \
		} \
		END{ \
			nm=0; for(k in mod){nm++;mk[nm]=k;mv[nm]=mod[k]} \
			for(i=2;i<=nm;i++){j=i;while(j>1&&mv[j]>mv[j-1]){ \
				t=mv[j];mv[j]=mv[j-1];mv[j-1]=t; \
				t=mk[j];mk[j]=mk[j-1];mk[j-1]=t;j--}} \
			printf "\n  %-34s %7s  %5s  %5s\n","Module","Lines","Files","%"; \
			printf "  %-34s %7s  %5s  %5s\n","------","-----","-----","--"; \
			for(i=1;i<=nm;i++) \
				printf "  %-34s %7d  %5d  %5.1f\n",mk[i],mv[i],fc[mk[i]],(total>0?mv[i]*100/total:0); \
			printf "  %-34s %7d  %5d\n","Total",total,files; \
			lm["Python"]=lang["py"]+0; \
			nl=0; for(k in lm) if(lm[k]>0){nl++;lk[nl]=k;lv[nl]=lm[k]} \
			for(i=2;i<=nl;i++){j=i;while(j>1&&lv[j]>lv[j-1]){ \
				t=lv[j];lv[j]=lv[j-1];lv[j-1]=t; \
				t=lk[j];lk[j]=lk[j-1];lk[j-1]=t;j--}} \
			printf "\n  %-34s %7s  %5s\n","Language","Lines","%"; \
			printf "  %-34s %7s  %5s\n","--------","-----","--"; \
			for(i=1;i<=nl;i++) \
				printf "  %-34s %7d  %5.1f\n",lk[i],lv[i],(total>0?lv[i]*100/total:0); \
			printf "\n" \
		}'

clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache .mypy_cache
	find . -name "*.tsbuildinfo" -type f -delete
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
