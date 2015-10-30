#!/bin/bash

if [[ $# < 2 ]] ; then
	echo 'USAGE : ./profile.sh [rPath] [VERSION]'
	exit 1
fi

PROFILE_REPORT=$PWD"/profile_report"
rPATH=$1
VERSION=$2

mkdir -p $PROFILE_REPORT/$VERSION

#all result
amplxe-cl -report hotspots -r $rPATH/ > \
		$PROFILE_REPORT/$VERSION/all

#compute_step result
amplxe-cl -report hotspots -source-object \
		function="Central2D<Shallow2D, MinMod<float>>::compute_step" > \
		$PROFILE_REPORT/$VERSION/compute_step

#limited_derivs result
amplxe-cl -report hotspots -source-object \
		function="Central2D<Shallow2D, MinMod<float>>::limited_derivs" > \
		$PROFILE_REPORT/$VERSION/limited_derivs

#compute_fg_speeds result
amplxe-cl -report hotspots -source-object \
		function="Central2D<Shallow2D, MinMod<float>>::compute_fg_speeds" > \
		$PROFILE_REPORT/$VERSION/compute_fg_speeds
