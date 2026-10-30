CADICALBUILD=/mnt/c/Users/Kushal Agarwal/Desktop/DAC-203 PROJECT/48_KAGGLE_OFFLOAD/src/cadical_stock/build
all:
	$(MAKE) -C "$(CADICALBUILD)"
clean:
	@if [ -d "$(CADICALBUILD)" ]; \
	then \
	  if [ -f "$(CADICALBUILD)"/makefile ]; \
	  then \
	     touch "$(CADICALBUILD)"/build.hpp; \
	     $(MAKE) -C "$(CADICALBUILD)" clean; \
	  fi; \
	  rm -rf "$(CADICALBUILD)"; \
	fi
	rm -f "/mnt/c/Users/Kushal Agarwal/Desktop/DAC-203 PROJECT/48_KAGGLE_OFFLOAD/src/cadical_stock/src/makefile"
	rm -f "/mnt/c/Users/Kushal Agarwal/Desktop/DAC-203 PROJECT/48_KAGGLE_OFFLOAD/src/cadical_stock/makefile"
test:
	$(MAKE) -C "$(CADICALBUILD)" test
cadical:
	$(MAKE) -C "$(CADICALBUILD)" cadical
mobical:
	$(MAKE) -C "$(CADICALBUILD)" mobical
update:
	$(MAKE) -C "$(CADICALBUILD)" update
format:
	$(MAKE) -C "$(CADICALBUILD)" format
.PHONY: all cadical clean mobical test format
