VLMRUN_SDK_VERSION := $(shell python -c 'from vlmrun.version import __version__; print(__version__.replace("-", "."))')
PYPI_USERNAME :=
PYPI_PASSWORD :=

WHL_GREP_PATTERN := .*\$(VLMRUN_SDK_VERSION).*\.whl

create-pypi-release-test:
	@echo "looking for vlmrun whl file..."
	@for file in dist/*; do \
		echo "examining file: $$file"; \
		if [ -f "$$file" ] && echo "$$file" | grep -qE "$(WHL_GREP_PATTERN)"; then \
			echo "Uploading: $$file"; \
			twine upload --repository testpypi "$$file"; \
		fi; \
	done
	@echo "Upload completed"


create-pypi-release:
	@echo "looking for vlmrun whl file..."
	@for file in dist/*; do \
		echo "examining file: $$file"; \
		if [ -f "$$file" ] && echo "$$file" | grep -qE "$(WHL_GREP_PATTERN)"; then \
			echo "Uploading: $$file"; \
			twine upload "$$file"; \
		fi; \
	done
	@echo "Upload completed"

create-tag:
	git tag -a ${VLMRUN_SDK_VERSION} -m "Release ${VLMRUN_SDK_VERSION}"
	git push origin ${VLMRUN_SDK_VERSION}
