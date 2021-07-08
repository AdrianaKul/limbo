
set(limbo_SOURCES
    ehvi/ehvi_calculations.cc
    ehvi/ehvi_montecarlo.cc
    ehvi/ehvi_sliceupdate.cc
    ehvi/ehvi_hvol.cc
    ehvi/ehvi_multi.cc
    ehvi/helper.cc
    hv/hypervol.c)

add_library(limbo ${limbo_SOURCES})
