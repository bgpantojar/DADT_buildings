#!/bin/bash
#FreeCAD path
export FREECADPATH=/home/pantojas/Bryan_PR2/04_TAPEDA_BP/04_UAV2EFM/09_DT_masonry_buildings/DADT_buildings/freecad_dev/usr/bin
#DADT_buildings path
export DADTPATH=/home/pantojas/Bryan_PR2/04_TAPEDA_BP/04_UAV2EFM/09_DT_masonry_buildings/DADT_buildings/src

$main_damage_file=$1
$main_DADT_file=$2

PYTHONPATH=${DADTPATH} PATH=$PATH:${FREECADPATH} python $main_damage_file $@

PYTHONPATH=${DADTPATH} PATH=$PATH:${FREECADPATH} freecadcmd $main_DADT_file $@

#How to run it: From terminal, being inside src folder, activate environment and run in terminal as:  
#./DADT.sh ../examples/p4_DADT_00_la_capite_main_damage.py ../examples/p4_DADT_00_la_capite_main_DADT.py
