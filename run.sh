#!/bin/bash
export LD_LIBRARY_PATH=/home/cyril/opt/ilmbase-2.2.0/lib:/home/cyril/opt/openexr-2.2.0/lib:$LD_LIBRARY_PATH
make
./denoise
/home/cyril/Installs/openexr_viewers-2.2.0/exrdisplay/exrdisplay test_image_dst.exr
/home/cyril/Installs/openexr_viewers-2.2.0/exrdisplay/exrdisplay test_image_src.exr
